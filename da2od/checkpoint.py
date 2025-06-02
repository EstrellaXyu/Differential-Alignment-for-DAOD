from typing import Any, Dict

from fvcore.common.checkpoint import _IncompatibleKeys
from detectron2.checkpoint.detection_checkpoint import DetectionCheckpointer
from detectron2.checkpoint.c2_model_loading import align_and_update_state_dicts


class CheckpointerWithEMA(DetectionCheckpointer):
    def __init__(self, model, save_dir="", *, save_to_disk=None, **checkpointables):
        super().__init__(model, save_dir=save_dir, save_to_disk=save_to_disk, **checkpointables)

    def resume_or_load(self, path: str, *, resume: bool = True) -> Dict[str, Any]:
        ret = super().resume_or_load(path, resume=resume)
        if (not resume) and path.endswith(".pth") and "ema" in ret.keys():
            self.logger.info("Loading EMA weights as model starting point.")
            ema_dict = {
                k.replace('model.',''): v for k, v in ret['ema'].items()
            }
            incompatible = self.model.load_state_dict(ema_dict, strict=False)
            if incompatible is not None:
                self._log_incompatible_keys(_IncompatibleKeys(
                    missing_keys=incompatible.missing_keys,
                    unexpected_keys=incompatible.unexpected_keys,
                    incorrect_shapes=[]
                ))
        return ret
    

