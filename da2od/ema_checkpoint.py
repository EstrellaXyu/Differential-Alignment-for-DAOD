from typing import Any, Dict
from collections import OrderedDict
import copy
from torch import nn

import detectron2.utils.comm as comm


from fvcore.common.checkpoint import _IncompatibleKeys
from detectron2.checkpoint.detection_checkpoint import DetectionCheckpointer
from detectron2.checkpoint.c2_model_loading import align_and_update_state_dicts

class EMA(nn.Module):
    def __init__(self, model, alpha):
        super(EMA, self).__init__()
        self.model = copy.deepcopy(model)
        self.alpha = alpha

    def get_stu_model_dict(self, model):
        if comm.get_world_size() > 1:
            student_model_dict = {
                key[7:]: value for key, value in model.state_dict().items()
            }
        else:
            student_model_dict = { k: v.to(self.model.device) for k,v in model.state_dict().items() }
        return student_model_dict

    def init_ema_model_weights(self, model):
        self.model.load_state_dict(self.get_stu_model_dict(model))

    def update_tea_model_dict(self, model, iter):
        student_model_dict = self.get_stu_model_dict(model)

        new_teacher_dict = OrderedDict()
        for key, value in self.model.state_dict().items():
            if key in student_model_dict.keys():
                new_teacher_dict[key] = (student_model_dict[key] *(1 - self.alpha) + value * self.alpha)
            else:
                raise Exception("{} is not found in student model".format(key))

        self.model.load_state_dict(new_teacher_dict)

    def update_weights(self, model, iter):
        # Init/update ema model
        if iter == 0:
            self.init_ema_model_weights(model)
        if iter > 0:
            self.update_tea_model_dict(model, iter)

    def inference(self, data, **kwargs):
        return self.model.inference(data, **kwargs)


class CheckpointerWithEMA(DetectionCheckpointer):
    def __init__(self, model, save_dir="", *, save_to_disk=None, **checkpointables):
        super().__init__(model, save_dir=save_dir, save_to_disk=save_to_disk, **checkpointables)

    def resume_or_load(self, path: str, *, resume: bool = True) -> Dict[str, Any]:
        checkpoint = super().resume_or_load(path, resume=resume)
        load_ema = (
            not resume
            and isinstance(path, str) 
            and path.endswith(".pth") 
            and "ema" in checkpoint
        )
        
        if load_ema:
            self.logger.info("Loading EMA weights as model starting point.")
            ema_dict = {
                k.replace('model.',''): v for k, v in checkpoint['ema'].items()
            }
            incompatible = self.model.load_state_dict(ema_dict, strict=False)
            if incompatible is not None:
                self._log_incompatible_keys(_IncompatibleKeys(
                    missing_keys=incompatible.missing_keys,
                    unexpected_keys=incompatible.unexpected_keys,
                    incorrect_shapes=[]
                ))
        return checkpoint


