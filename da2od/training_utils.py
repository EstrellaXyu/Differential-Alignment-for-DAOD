import logging
import weakref
import time
import torch
import copy
import numpy as np

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data.dataset_mapper import DatasetMapper as _DatasetMapper
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.engine.train_loop import TrainerBase
from detectron2.engine.train_loop import AMPTrainer as _AMPTrainer
from detectron2.engine.train_loop import SimpleTrainer as _SimpleTrainer
from detectron2.engine.defaults import create_ddp_model
from detectron2.engine.defaults import DefaultTrainer as _DefaultTrainer
from detectron2.utils.logger import setup_logger
from detectron2.utils import comm
from detectron2.utils.events import get_event_storage

import random
import torch
import numpy as np
import os
import cv2

from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

class SaveIO:
    """Simple PyTorch hook to save the output of a nn.module."""
    def __init__(self):
        self.input = None
        self.output = None
        
    def __call__(self, module, module_in, module_out):
        self.input = module_in
        self.output = module_out

class ManualSeed:
    """PyTorch hook to manually set the random seed."""
    def __init__(self):
        self.reset_seed()

    def reset_seed(self):
        self.seed = random.randint(0, 2**32 - 1)

    def __call__(self, module, args):
        torch.manual_seed(self.seed)

class ReplaceProposalsOnce:
    """PyTorch hook to replace the proposals with the student's proposals, but only once."""
    def __init__(self):
        self.proposals = None

    def set_proposals(self, proposals):
        self.proposals = proposals

    def __call__(self, module, args):
        ret = None
        if self.proposals is not None and module.training:
            images, features, proposals, gt_instances = args
            ret = (images, features, self.proposals, gt_instances)
            self.proposals = None
        return ret

def set_attributes(obj, params):
    """Set attributes of an object from a dictionary."""
    if params:
        for k, v in params.items():
            if k != "self" and not k.startswith("_"):
                setattr(obj, k, v)

class _GradientScalarLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.weight = weight
        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return ctx.weight*grad_input, None

def grad_reverse(x):
    return _GradientScalarLayer.apply(x, -1.0)

def _maybe_add_optional_annotations(cocoapi) -> None:
    for ann in cocoapi.dataset["annotations"]:
        if "iscrowd" not in ann:
            ann["iscrowd"] = 0
        if "area" not in ann:
            ann["area"] = ann["bbox"][1]*ann["bbox"][2]

class Detectron2COCOEvaluatorAdapter(COCOEvaluator):
    """A COCOEvaluator that makes iscrowd & area optional."""
    def __init__(
        self,
        dataset_name,
        output_dir=None,
        distributed=True,
    ):
        super().__init__(dataset_name, output_dir=output_dir, distributed=distributed)
        _maybe_add_optional_annotations(self._coco_api)

class CustomCOCOEvaluator(COCOEvaluator):
    def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
        # Call the original method to get the standard metrics
        results = super()._derive_coco_results(coco_eval, iou_type, class_names)

        # Extract AP50
        ap50 = coco_eval.stats[1]  # AP@[ IoU=0.50 ]
        results["AP50"] = ap50

        # If class names are provided, compute AP50 for each class
        if class_names:
            precisions = coco_eval.eval['precision']
            assert len(class_names) == precisions.shape[2]

            for idx, name in enumerate(class_names):
                precision = precisions[0, :, idx, 0, 2]
                results[f'{name}_AP50'] = float(np.mean(precision[precision > -1]))
        return results

class DefaultTrainer(_DefaultTrainer):
    """
    Same as detectron2.engine.defaults.DefaultTrainer, but adds:
     - a _create_trainer method to allow easier use of other trainers.
     - a _create_checkpointer method to allow for overwriting custom checkpointer logic
    """
    def __init__(self, cfg):
            """
            Args:
                cfg (CfgNode):
            """
            # call grandparent init, so we can overwrite the DefaultTrainer __init__ logic completely
            TrainerBase.__init__(self) 
            
            logger = logging.getLogger("detectron2")
            if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
                setup_logger()
            cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

            model = self.build_model(cfg)
            optimizer = self.build_optimizer(cfg, model)
            data_loader = self.build_train_loader(cfg)

            ### Change is here ###
            model = self.create_ddp_model(model, broadcast_buffers=False, cfg=cfg)
            ###   End change   ###

            ## Change is here ##
            self._trainer = self._create_trainer(cfg, model, data_loader, optimizer)
            ##   End change   ##

            self.scheduler = self.build_lr_scheduler(cfg, optimizer)

            ## Change is here ##
            self.checkpointer = self._create_checkpointer(model, cfg)
            ##   End change   ##

            self.start_iter = 0
            self.max_iter = cfg.SOLVER.MAX_ITER
            self.cfg = cfg

            self.register_hooks(self.build_hooks())

    def _create_trainer(self, cfg, model, data_loader, optimizer):
         return (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
                model, data_loader, optimizer
            )
    
    def _create_checkpointer(self, model, cfg, ckpt_cls=DetectionCheckpointer):
        return ckpt_cls(
                model,
                cfg.OUTPUT_DIR,
                trainer=weakref.proxy(self),
            )

    def create_ddp_model(self, model, broadcast_buffers, cfg):
        return create_ddp_model(model, broadcast_buffers=broadcast_buffers)
    
class SimpleTrainer(_SimpleTrainer):
    """
    Same as detectron2.engine.train_loop.SimpleTrainer, but:
    - adds a run_model method that provides a way to change the model's forward pass without 
    having to copy-paste the entire run_step method.
    - adds a do_backward method that provides a way to modify the backward pass
    """
    def run_step(self):
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        if self.zero_grad_before_forward:
            self.optimizer.zero_grad()

        ## Change is here ##
        loss_dict = self.run_model(data)
        ##   End change   ##

        if isinstance(loss_dict, torch.Tensor):
            losses = loss_dict
            loss_dict = {"total_loss": loss_dict}
        else:
            losses = sum(loss_dict.values())
        if not self.zero_grad_before_forward:
            self.optimizer.zero_grad()

        ## Change is here ##
        self.do_backward(losses)
        ##   End change   ##

        self.after_backward()
        self._write_metrics(loss_dict, data_time)
        self.optimizer.step()
    
    def run_model(self, data):
        return self.model(data)
    
    def do_backward(self, losses):
        losses.backward()

class AMPTrainer(_AMPTrainer):
    """
    Same as detectron2.engine.train_loop.AMPTrainer, but:
    - adds a run_model method that provides a way to change the model's forward pass without 
    having to copy-paste the entire run_step method.
    - adds a do_backward method that provides a way to modify the backward pass
    """
    def run_step(self):
        """
        Implement the AMP training logic.
        """
        assert self.model.training, "[AMPTrainer] model was changed to eval mode!"
        assert torch.cuda.is_available(), "[AMPTrainer] CUDA is required for AMP training!"
        from torch.cuda.amp import autocast

        start = time.perf_counter()
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        if self.zero_grad_before_forward:
            self.optimizer.zero_grad()
        with autocast(dtype=self.precision):
            
            ## Change is here ##
            loss_dict = self.run_model(data)
            ##   End change   ##

            if isinstance(loss_dict, torch.Tensor):
                losses = loss_dict
                loss_dict = {"total_loss": loss_dict}
            else:
                losses = sum(loss_dict.values())

        if not self.zero_grad_before_forward:
            self.optimizer.zero_grad()

        ## Change is here ##
        self.do_backward(losses)
        ##   End change   ##

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
    
class DatasetMapper(_DatasetMapper):
    def __call__(self, dataset_dict):
        """
        Same as detectron2.data.dataset_mapper.DatasetMapper, but adds a way to
        access the aug_input object in subclasses without copy-pasting the entire
        __call__ method.
        """
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
        else:
            sem_seg_gt = None
        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        transforms = self.augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg
        image_shape = image.shape[:2]  # h, w
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))
        if self.proposal_topk is not None:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, proposal_topk=self.proposal_topk
            )
        if not self.is_train:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict
        if "annotations" in dataset_dict:
            self._transform_annotations(dataset_dict, transforms, image_shape)

        ## Change is here ##
        dataset_dict = self._after_call(dataset_dict, aug_input)
        ##   End change   ##

        return dataset_dict
    
    def _after_call(self, dataset_dict, aug_input):
        return dataset_dict