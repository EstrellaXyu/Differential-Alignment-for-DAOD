import torch
from typing import Dict, List, Optional

from detectron2.config import configurable
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.utils.logger import _log_api_usage


from da2od.aligner_pred_guided import AlignMixin
from da2od.distiller_return_pred_diff import DistillMixin


def build_da2od(cfg):
    """Add Align and Distill capabilities to any Meta Architecture dynamically."""
    base_cls = META_ARCH_REGISTRY.get(cfg.MODEL.META_ARCHITECTURE)

    class DA2OD(AlignMixin, DistillMixin, base_cls):    
        @configurable
        def __init__(self, **kwargs):
            super(DA2OD, self).__init__(**kwargs)

        @classmethod
        def from_config(cls, cfg):
            return super(DA2OD, cls).from_config(cfg)

        # # change here
        # def forward(self, batched_inputs: List[Dict[str, torch.Tensor]], 
        #             alpha: float=1.0, do_align: bool = False, pred_diff: Optional[torch.Tensor] = None):
        #     return super(DA2OD, self).forward(batched_inputs, do_align=do_align, alpha=alpha, pred_diff=pred_diff) # 这里pred_diff还有值
        # # end change
        
        def forward(self, batched_inputs: List[Dict[str, torch.Tensor]], 
                    alpha: float=1.0, do_align: bool = False, pred_diff: Optional[torch.Tensor] = None, pseudo_boxes:  Optional[torch.Tensor] = None):
            return super(DA2OD, self).forward(batched_inputs, do_align=do_align, alpha=alpha, pred_diff=pred_diff, pseudo_boxes=pseudo_boxes) # 这里pred_diff还有值
        
    model = DA2OD(cfg)
    model.to(torch.device(cfg.MODEL.DEVICE))
    _log_api_usage("modeling.meta_arch." + cfg.MODEL.META_ARCHITECTURE)
    return model