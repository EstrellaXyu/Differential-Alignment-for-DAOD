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
# class CustomCOCOEvaluator(COCOEvaluator):
#     def __init__(self, dataset_name, cfg, distributed, output_dir=None):
#         super().__init__(dataset_name, cfg, distributed, output_dir)
#         self.metadata = MetadataCatalog.get(dataset_name)
#         self.cfg = cfg

#     def process(self, inputs, outputs):
#         super().process(inputs, outputs)

#         for input, output in zip(inputs, outputs):
#             image = input["image"].permute(1, 2, 0).cpu().numpy()
#             v = Visualizer(image, self.metadata, scale=1.2)
#             instances = output["instances"].to("cpu")
#             vis_output = v.draw_instance_predictions(instances).get_image()

#             # Save the visualized output
#             file_name = os.path.basename(input["file_name"])
#             output_file = os.path.join(self.cfg.OUTPUT_DIR, "visualizations", file_name)
#             os.makedirs(os.path.dirname(output_file), exist_ok=True)
#             cv2.imwrite(output_file, vis_output[:, :, ::-1])