import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import random
import math

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
import torch
from fvcore.transforms.transform import NoOpTransform, Transform
from torchvision.transforms import transforms as tv_transforms

from detectron2.config import CfgNode, configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.transforms.augmentation import Augmentation, _get_aug_input_args
from detectron2.data.transforms.augmentation_impl import RandomApply


logger = logging.getLogger(__name__)
WEAK_IMG_KEY = "img_weak"

#-------------Start Define Customized Transformation----------#
class WeakImageSaver(T.Augmentation):
    def __init__(self, savename):
        super().__init__()
        self._init(locals())

    def get_transform(self, image: np.ndarray) -> Transform:
        return NoOpTransform()

    def __call__(self, aug_input) -> Transform:
        image = _get_aug_input_args(self, aug_input)[0].copy()
        setattr(aug_input, self.key, image)
        return super().__call__(aug_input)

class BaseTransform(Transform):
    def apply_image(self, img:np.array) -> np.array:
        return img.copy()
    
    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        return segmentation

    def inverse(self) -> Transform:
        return NoOpTransform()

class GaussianBlurTransform(BaseTransform):
    def __init__(self, sigma_range: Tuple[float, float]):
        super().__init__()
        self.sigma_min, self.sigma_max = sigma_range

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        sigma = np.random.uniform(self.sigma_min, self.sigma_max)
        if img.dtype == np.uint8:
            img = img.astype(np.float32)
            img = gaussian_filter(img, sigma=sigma)
            return np.clip(img, 0, 255).astype(np.uint8)
        else:
            return gaussian_filter(img, sigma=sigma)

class RandomErasing(BaseTransform):
    def __init__(self, scale_range: Tuple[float, float], ratio_range: Tuple[float, float], value: Union[str, list] = [0.4914, 0.4822, 0.4465]):
        super().__init__()
        self.scale_range = scale_range
        self.ratio_range = ratio_range
        self.value = value

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        is_uint8 = img.dtype == np.uint8
        if is_uint8:
            img = img.astype(np.float32)
        H, W, C = img.shape
        area = H * W

        for _ in range(100):
            target_area = np.random.uniform(*self.scale_range) * area
            aspect_ratio = np.random.uniform(*self.ratio_range)

            h = int(round((target_area * aspect_ratio) ** 0.5))
            w = int(round((target_area / aspect_ratio) ** 0.5))

            if w>1 and h>1 and h < H and w < W:
                w0 = np.random.randint(0, W - w - 1)
                h0 = np.random.randint(0, H - h - 1)
                
                if self.value == "random":
                    img[h0:h0+h, w0:w0+w, :] = np.random.rand(h, w, C)
                else:
                    img[h0:h0+h, w0:w0+w, 0] = self.value[0]
                    img[h0:h0+h, w0:w0+w, 1] = self.value[1]
                    img[h0:h0+h, w0:w0+w, 2] = self.value[2]
                break
        if is_uint8:
            img[h0:h0+h, w0:w0+w, :] *= 255
            return np.clip(img, 0, 255).astype(np.uint8)
        else:
            return img

class MICTransform(BaseTransform):
    """MIC masking transform"""
    def __init__(self, ratio: float, block_size: int):
        super().__init__()
        self.ratio = ratio
        self.block_size = block_size

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        H, W, C = img.shape
        is_uint8 = img.dtype == np.uint8
        
        if is_uint8:
            img = img.astype(np.float32)

        h_blocks = max(1, round(H / self.block_size))
        w_blocks = max(1, round(W / self.block_size))
        
        mask = np.random.rand(h_blocks, w_blocks) > self.ratio
        mask = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        
        img = img * np.repeat(mask[..., None], C, axis=-1)
        
        if is_uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        else:
            return img     
#-------------End Define Customized Transformation----------#

#---------Start Define Weak / Strong Augmentations Set------#
class AugmentationBuilder:
    @configurable
    def __init__(self, is_train: bool, weak_aug_cfg: dict, strong_aug_cfg: dict):
        self.is_train = is_train
        self.weak_aug_cfg = weak_aug_cfg
        self.strong_aug_cfg = strong_aug_cfg

    @classmethod
    def from_config(cls, cfg):
        return cls(
            is_train=True,
            weak_aug_cfg = {
                "enable_crop": cfg.INPUT.CROP.ENABLED,
                "crop_type": cfg.INPUT.CROP.TYPE,
                "crop_size": cfg.INPUT.CROP.SIZE
            },
            strong_aug_cfg = {
                "color_jitter": {
                    "brightness": (0.6, 1.4),
                    "contrast": (0.6, 1.4),
                    "saturation": (0.6, 1.4),
                },
                "blur": {"sigma": (0.1, 2.0)},
                "erase": [
                    {"scale": (0.05, 0.2), "ratio": (0.3, 3.3), "prob": 0.7},
                    {"scale": (0.02, 0.2), "ratio": (0.1, 6.0), "prob": 0.5},
                    {"scale": (0.02, 0.2), "ratio": (0.05, 8.0), "prob": 0.3}
                ]
            }
        )

    def build_weak_augmentation(self, cfg) -> List[Transform]:
        augs = utils.build_augmentation(cfg, is_train=True)

        # import pdb; pdb.set_trace()
        if self.weak_aug_cfg["enable_crop"]:
            augs.insert(0, T.RandomCrop(self.weak_aug_cfg["crop_type"], self.weak_aug_cfg["crop_size"]))
            
        augs.append(WeakImageSaver(WEAK_IMG_KEY))
        return augs

    def build_strong_augmentation(self, include_erasing: bool = True) -> List[Transform]:
        cfg = self.strong_aug_cfg
        augs = [
            T.RandomApply(
                T.AugmentationList([
                    T.RandomContrast(*cfg["color_jitter"]["contrast"]),
                    T.RandomBrightness(*cfg["color_jitter"]["brightness"]),
                    T.RandomSaturation(*cfg["color_jitter"]["saturation"])
                ]),
                prob=0.8
            ),
            T.RandomApply(T.RandomSaturation(0, 0), prob=0.2),
            T.RandomApply(GaussianBlurTransform(cfg["blur"]["sigma"]), prob=0.5)
        ]

        if include_erasing:
            for erase_cfg in cfg["erase"]:
                augs.append(
                    T.RandomApply(
                        RandomErasing(scale_range=erase_cfg["scale"], ratio_range=erase_cfg["ratio"], value="random"),
                            prob=erase_cfg["prob"])
                )

        return augs

def build_augmentation(cfg: CfgNode, is_labeled: bool, include_strong: bool = True) -> List[Transform]:
    builder = AugmentationBuilder.from_config(cfg)
    augs = builder.build_weak_augmentation(cfg)

    if include_strong:
        use_erasing = (is_labeled and cfg.AUG.LABELED_INCLUDE_RANDOM_ERASING) or \
                    (not is_labeled and cfg.AUG.UNLABELED_INCLUDE_RANDOM_ERASING)
        augs.append(builder.build_strong_augmentation(include_erasing=use_erasing))

        including_mic = (is_labeled and cfg.AUG.LABELED_MIC_AUG) or \
                (not is_labeled and cfg.AUG.UNLABELED_MIC_AUG)
        if including_mic:
            augs.append(
                T.RandomApply(
                    MICTransform(ratio=cfg.AUG.MIC_RATIO, block_size=cfg.AUG.MIC_BLOCK_SIZE),
                    prob=1.0
                )
            )

    return augs
#---------End Define Weak / Strong Augmentations Set------#
