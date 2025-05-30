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
        return create_ddp_model(model, broadcast_buffers=broadcast_buffers, find_unused_parameters=True)
    
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