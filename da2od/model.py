import torch
from typing import Dict, List, Optional

from detectron2.utils.logger import _log_api_usage
from detectron2.utils.registry import Registry

from detectron2.config import configurable
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY

from da2od.aligner_pred_guided import DA2ODAligner
from da2od.distiller_return_pred_diff import DA2ODDistiller, DistillMixin

def build_da2od(cfg):
    det_model = META_ARCH_REGISTRY.get(cfg.MODEL.META_ARCHITECTURE)

    class DA2OD(DA2ODAligner, DistillMixin, det_model):    
        @configurable
        def __init__(self, **kwargs):
            super(DA2OD, self).__init__(**kwargs)

        @classmethod
        def from_config(cls, cfg):
            return super(DA2OD, cls).from_config(cfg)

        def forward(self, batched_inputs: List[Dict[str, torch.Tensor]], 
                    do_align: bool = False,
                    domain_label: float=1.0, 
                    pred_diff: Optional[torch.Tensor] = None, 
                    pseudo_boxes:  Optional[torch.Tensor] = None):
            return super(DA2OD, self).forward(batched_inputs, 
                                              do_align=do_align, 
                                              domain_label=domain_label, 
                                              pred_diff=pred_diff, 
                                              pseudo_boxes=pseudo_boxes)
    
    model = DA2OD(cfg)
    model.to(torch.device(cfg.MODEL.DEVICE))
    _log_api_usage("modeling.meta_arch." + cfg.MODEL.META_ARCHITECTURE)
    return model