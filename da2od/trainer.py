import os
import time
import weakref
import copy
import logging
import torch
import numpy as np
import random
from torch.nn.parallel import DistributedDataParallel as DDP


from detectron2.engine.train_loop import TrainerBase
from detectron2.engine.defaults import DefaultTrainer, create_ddp_model
from detectron2.utils.logger import setup_logger

from detectron2.checkpoint.detection_checkpoint import DetectionCheckpointer
from detectron2.data.build import build_detection_train_loader, get_detection_dataset_dicts
from detectron2.engine import hooks, BestCheckpointer
from detectron2.evaluation import DatasetEvaluators
from detectron2.solver import build_optimizer
from detectron2.utils.events import get_event_storage
from detectron2.utils import comm
from detectron2.engine.train_loop import AMPTrainer, SimpleTrainer

from da2od.model import build_da2od

from da2od.aug_modi import build_augmentation
from da2od.ema_checkpoint import CheckpointerWithEMA, EMA
from da2od.distiller_return_pred_diff import build_distiller
from da2od.model import build_da2od
from da2od.utils import CustomCOCOEvaluator
from da2od.dataloader import UnlabeledMapper, SemiDataloader, InterWeakSaveMapper


#--------------Start trainer about Single training loop---------------#

class _SimpleTrainer(SimpleTrainer):
     def run_step(self):
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        if self.zero_grad_before_forward:
            self.optimizer.zero_grad()

        loss_dict = self.run_model(data)

        if isinstance(loss_dict, torch.Tensor):
            losses = loss_dict
            loss_dict = {"total_loss": loss_dict}
        else:
            losses = sum(loss_dict.values())
        if not self.zero_grad_before_forward:
            self.optimizer.zero_grad()

        self.do_backward(losses)

        self.after_backward()
        self._write_metrics(loss_dict, data_time)
        self.optimizer.step()
    
     def run_model(self, data):
          return self.model(data)
     
     def do_backward(self, losses):
          losses.backward()

class _AMPTrainer(AMPTrainer):
     def run_step(self):
        assert self.model.training, "[AMPTrainer] model was changed to eval mode!"
        assert torch.cuda.is_available(), "[AMPTrainer] CUDA is required for AMP training!"
        from torch.cuda.amp import autocast

        start = time.perf_counter()
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        if self.zero_grad_before_forward:
            self.optimizer.zero_grad()
        with autocast(dtype=self.precision):
            loss_dict = self.run_model(data)

            if isinstance(loss_dict, torch.Tensor):
                losses = loss_dict
                loss_dict = {"total_loss": loss_dict}
            else:
                losses = sum(loss_dict.values())

        if not self.zero_grad_before_forward:
            self.optimizer.zero_grad()

        self.do_backward(losses)

        if self.log_grad_scaler:
            storage = get_event_storage()
            storage.put_scalar("[metric]grad_scaler", self.grad_scaler.get_scale())

        self.after_backward()

        self._write_metrics(loss_dict, data_time)

        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

     def run_model(self, data):
          return self.model(data)
     
     def do_backward(self, losses):
          self.grad_scaler.scale(losses).backward()

class DA2ODBaseTrainer:
     def __init__(self, model, data_loader, optimizer, distiller, backward_every_phase=False, model_batch_size=None):
          super().__init__(model, data_loader, optimizer, zero_grad_before_forward=backward_every_phase)
          self.distiller = distiller
          self.backward_every_phase = backward_every_phase
          self.model_batch_size = model_batch_size

     def run_model(self, data):
          return da2od_run_model(self, *data)
     
     def do_backward(self, losses, override=False):
        """Disable the final backward pass if we are computing intermediate gradients in.
        Can be overridden by setting override=True to always call superclass method."""
        if not self.backward_every_phase or override:
             super().do_backward(losses)
             
class DA2ODAMPTrainer(DA2ODBaseTrainer, _AMPTrainer): pass
class DA2ODSimpleTrainer(DA2ODBaseTrainer, _SimpleTrainer): pass

def da2od_run_model(trainer, labeled_weak, labeled_strong, unlabeled_weak, unlabeled_strong):
          """
          - Warm up phase: using labeled images only
          - Mutual learning phase: using teacher-student framework, e.g. differential aligning and distilling
          """
          def do_backward_every_phase(losses, backward_condition=lambda k: True):
               if backward_every_phase:
                    losses = {k: v * 0 if not backward_condition(k) else v for k, v in losses.items() }
                    trainer.do_backward(sum(losses.values()) / num_grad_accum_steps, override=True)
          
          def merge_loss_dict(losses, suffix, key_conditional=lambda k: True):
               for k, v in losses.items():
                    if key_conditional(k):
                         v /= num_grad_accum_steps
                         if backward_every_phase: 
                              v = v.detach()
                         loss_dict[f"{k}_{suffix}"] = loss_dict.get(f"{k}_{suffix}", 0) + v
               
          def do_training_step(data, data_name="", backward_condition=lambda k: True, **kwargs):
               for batch_i in range(0, len(data), model_batch_size):
                    loss = model(data[batch_i:batch_i+model_batch_size], **kwargs)
                    do_backward_every_phase(loss, backward_condition)
                    merge_loss_dict(loss, data_name, backward_condition)

          def do_distill_step(teacher_data, student_data, name="", key_condition=lambda k: True, **kwargs):
               assert len(teacher_data) == len(student_data), "Teacher and student data must be the same length."
               for batch_i in range(0, len(teacher_data), model_batch_size):
                    distill_loss, pred_diff, pseudo_boxes = trainer.distiller(teacher_data[batch_i:batch_i+model_batch_size], 
                                                  student_data[batch_i:batch_i+model_batch_size])
                    do_backward_every_phase(distill_loss, key_condition)
                    merge_loss_dict(distill_loss, name, key_condition)
                    return pred_diff, pseudo_boxes

          model = trainer.model
          _model = model.module if type(model) == DDP else model
          
          backward_every_phase = trainer.backward_every_phase
          model_batch_size = trainer.model_batch_size 
          total_batch_size = sum([len(s or []) for s in [labeled_strong, unlabeled_weak, unlabeled_strong]])
          num_grad_accum_steps = total_batch_size // model_batch_size
          
          do_labeled_weak = labeled_weak is not None
          do_labeled_strong = labeled_strong is not None
          do_align = any( [ getattr(_model, a, None) is not None for a in ["img_align", "ins_align"] ] )
          do_distill = trainer.distiller.is_distill_enabled()
          
          loss_dict = {}
          if do_labeled_weak:
               do_training_step(labeled_weak, "source_weak", lambda k: do_labeled_weak or (do_align and "_da_" in k), do_align=do_align)
          if do_labeled_strong:
               do_training_step(labeled_strong, "source_strong", lambda k: do_labeled_strong or (do_align and "_da_" in k), do_align=do_align)
          if do_distill:
               pred_diff, pseudo_boxes = do_distill_step(unlabeled_weak, unlabeled_strong, "distill", lambda k: k != "_")
               if do_align:
                    do_training_step(unlabeled_weak, "target_weak", lambda k: "_da_" in k, domain_label=0.0, do_align=True)    # domain_label = 1 if labeded else 0
                    do_training_step(unlabeled_strong, "target_strong", lambda k: "_da_" in k, domain_label=0.0, do_align=True, pred_diff=pred_diff, pseudo_boxes=pseudo_boxes)
          
          return loss_dict
          
#------------------End trainer about training loop--------------------#

#--------------Start trainer about whole training components----------#
class SemiBaseTrainer(DefaultTrainer):
    def __init__(self, cfg):
            TrainerBase.__init__(self) 
            
            logger = logging.getLogger("detectron2")
            if not logger.isEnabledFor(logging.INFO): 
                setup_logger()
            cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

            model = self.build_model(cfg)
            optimizer = self.build_optimizer(cfg, model)
            data_loader = self.build_train_loader(cfg)
            
            # add find_unused_parameters to avoid warning
            model = create_ddp_model(model, broadcast_buffers=False, find_unused_parameters=True)

            self._trainer = self.create_trainer(cfg, model, data_loader, optimizer)

            self.scheduler = self.build_lr_scheduler(cfg, optimizer)

            self.checkpointer = self.create_checkpointer(model, cfg)

            self.start_iter = 0
            self.max_iter = cfg.SOLVER.MAX_ITER
            self.cfg = cfg

            self.register_hooks(self.build_hooks())

    def create_trainer(self, cfg, model, data_loader, optimizer):
         return (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
                model, data_loader, optimizer
            )
    
    def create_checkpointer(self, model, cfg, ckpt_cls=DetectionCheckpointer):
        return ckpt_cls(
                model,
                cfg.OUTPUT_DIR,
                trainer=weakref.proxy(self),
            )

class DA2ODTrainer(SemiBaseTrainer):
     @classmethod
     def build_train_loader(cls, cfg):
          batch_components = cfg.DATASETS.BATCH_COMPONENTS
          batch_ratios = cfg.DATASETS.BATCH_RATIOS
          total_batch_size = cfg.SOLVER.IMS_PER_BATCH
          label_unlabeded_batch_sizes = [ int(total_batch_size * r / sum(batch_ratios)) for r in batch_ratios ]
          assert len(batch_components) == len(label_unlabeded_batch_sizes), "len(cfg.DATASETS.batch_components) must equal len(cfg.DATASETS.BATCH_RATIOS)."

          labeled_bs = [label_unlabeded_batch_sizes[i] for i in range(len(batch_components)) if batch_components[i].startswith("labeled")]
          labeled_bs = max(labeled_bs) if len(labeled_bs) else 0
          unlabeled_bs = [label_unlabeded_batch_sizes[i] for i in range(len(batch_components)) if batch_components[i].startswith("unlabeled")]
          unlabeled_bs = max(unlabeled_bs) if len(unlabeled_bs) else 0
          
          # create labeled dataloader
          labeled_loader = None
          if labeled_bs > 0 and len(cfg.DATASETS.TRAIN):
               labeled_loader = build_detection_train_loader(get_detection_dataset_dicts(cfg.DATASETS.TRAIN, filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS), 
                    mapper=InterWeakSaveMapper(cfg, is_train=True, augmentations=build_augmentation(cfg, labeled=True, include_strong_augs="labeled_strong" in batch_components)),
                    num_workers=cfg.DATALOADER.NUM_WORKERS, 
                    total_batch_size=labeled_bs)

          # create unlabeled dataloader
          unlabeled_loader = None
          if unlabeled_bs > 0 and len(cfg.DATASETS.UNLABELED):
               unlabeled_loader = build_detection_train_loader(get_detection_dataset_dicts(cfg.DATASETS.UNLABELED, filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS), 
                    mapper=UnlabeledMapper(cfg, is_train=True, augmentations=build_augmentation(cfg, labeled=False, include_strong_augs="unlabeled_strong" in batch_components)),
                    num_workers=cfg.DATALOADER.NUM_WORKERS,
                    total_batch_size=unlabeled_bs)

          return SemiDataloader(labeled_loader, unlabeled_loader, batch_components)

     @classmethod
     def build_model(cls, cfg):
          model = build_da2od(cfg)
          logger = logging.getLogger(__name__)
          logger.info("Model:\n{}".format(model))
          return model
     
     @classmethod
     def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return CustomCOCOEvaluator(dataset_name, cfg, True, output_folder)
          
     def create_trainer(self, cfg, model, data_loader, optimizer):
          self.ema = EMA(build_da2od(cfg), cfg.EMA.ALPHA) if cfg.EMA.ENABLED else None
          distiller = build_distiller(teacher=self.ema.model if cfg.EMA.ENABLED else model, student=model, cfg=cfg)
          trainer = (DA2ODAMPTrainer if cfg.SOLVER.AMP.ENABLED else DA2ODSimpleTrainer)(model, data_loader, optimizer, distiller, 
                                                                                        backward_every_phase=cfg.SOLVER.BACKWARD_EVERY_PHASE,
                                                                                        model_batch_size=cfg.SOLVER.IMS_PER_GPU)
          return trainer
     
     def create_checkpointer(self, model, cfg):
          checkpointer = super(DA2ODTrainer, self).create_checkpointer(model, cfg, 
                         ckpt_cls=CheckpointerWithEMA if cfg.EMA.LOAD_FROM_EMA_ON_START else DetectionCheckpointer)
          if cfg.EMA.ENABLED:
               checkpointer.add_checkpointable("ema", self.ema)
          return checkpointer
     
     def before_step(self):
          """Update the EMA model every step."""
          super(DA2ODTrainer, self).before_step()
          if self.cfg.EMA.ENABLED:
               self.ema.update_weights(self._trainer.model, self.iter)
               
     def build_hooks(self):
          ret = super(DA2ODTrainer, self).build_hooks()

          # add hooks to evaluate/save teacher model if applicable
          if self.cfg.EMA.ENABLED:
               def test_and_save_results_ema():
                    self._last_eval_results = self.test(self.cfg, self.ema.model)
                    return self._last_eval_results
               eval_hook = hooks.EvalHook(self.cfg.TEST.EVAL_PERIOD, test_and_save_results_ema)
               if comm.is_main_process():
                    ret.insert(-1, eval_hook) # before PeriodicWriter if in main process
               else:
                    ret.append(eval_hook)

          # add a hook to save the best (teacher, if EMA enabled) checkpoint to model_best.pth
          if comm.is_main_process():
               if len(self.cfg.DATASETS.TEST) == 1:
                    ret.insert(-1, BestCheckpointer(self.cfg.TEST.EVAL_PERIOD, self.checkpointer,
                                                    f"bbox/AP50", "max", file_prefix=f"{self.cfg.DATASETS.TEST[0]}_model_best"))
               else: 
                    for test_set in self.cfg.DATASETS.TEST:
                         ret.insert(-1, BestCheckpointer(self.cfg.TEST.EVAL_PERIOD, self.checkpointer,
                                                    f"{test_set}/bbox/AP50", "max", file_prefix=f"{test_set}_model_best"))
          return ret

#--------------End trainer about whole training components----------#