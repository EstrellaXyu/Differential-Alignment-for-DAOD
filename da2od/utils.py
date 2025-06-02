import torch
import random
import os
from detectron2.data import MetadataCatalog
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.visualizer import Visualizer
import cv2

class CustomCOCOEvaluator(COCOEvaluator):
    def __init__(self, dataset_name, cfg, distributed, output_dir=None):
        super().__init__(dataset_name, cfg, distributed, output_dir)
        self.metadata = MetadataCatalog.get(dataset_name)
        self.cfg = cfg

    def process(self, inputs, outputs):
        super().process(inputs, outputs)

        for input, output in zip(inputs, outputs):
            image = input["image"].permute(1, 2, 0).cpu().numpy()
            v = Visualizer(image, self.metadata, scale=1.2)
            instances = output["instances"].to("cpu")
            vis_output = v.draw_instance_predictions(instances).get_image()

            # Save the visualized output
            file_name = os.path.basename(input["file_name"])
            output_file = os.path.join(self.cfg.OUTPUT_DIR, "visualizations", file_name)
            if file_name.split('_')[-1] != "leftImg8bit.png":
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                cv2.imwrite(output_file, vis_output[:, :, ::-1])

class SaveIO:
    """Simple PyTorch hook to save the output of a nn.module."""
    def __init__(self):
        self.input = None
        self.output = None
        
    def __call__(self, module, module_in, module_out):
        self.input = module_in
        self.output = module_out

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

def set_attributes(obj, params):
    """Set attributes of an object from a dictionary."""
    if params:
        for k, v in params.items():
            if k != "self" and not k.startswith("_"):
                setattr(obj, k, v)

class RandomSeed:
    def __init__(self):
        self.reset_seed()

    def reset_seed(self):
        self.seed = random.randint(0, 2**32 - 1)

    def __call__(self, module, args):
        torch.manual_seed(self.seed)
