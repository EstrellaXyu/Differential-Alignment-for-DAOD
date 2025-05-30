import copy
from typing import Optional, List, Dict, Any, Tuple, Iterator
import torch
import numpy as np
from detectron2.data import DatasetMapper
from detectron2.structures import Instances, Boxes
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from da2od.aug_modi import WEAK_IMG_KEY
import itertools

#----------Start Define Labeled / Unlabeled Dataset Mapper----------#
class DatasetMapper_after_call(DatasetMapper):
    def __call__(self, dataset_dict):
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

        dataset_dict = self._after_call(dataset_dict, aug_input)

        return dataset_dict
    
    def _after_call(self, dataset_dict, aug_input):
        return dataset_dict

class InterWeakSaveMapper(DatasetMapper_after_call):   
    def process_weak_augmentation(self, dataset_dict, aug_input):
        weak_img = getattr(aug_input, WEAK_IMG_KEY)
        dataset_dict[WEAK_IMG_KEY] = torch.as_tensor(np.ascontiguousarray(weak_img.transpose(2, 0, 1)))
        return dataset_dict
    
    def _after_call(self, dataset_dict, aug_input):
        return self.process_weak_augmentation(dataset_dict, aug_input)

class UnlabeledMapper(InterWeakSaveMapper):
    def __call__(self, dataset_dict):
        dataset_dict = super().__call__(dataset_dict)

        dataset_dict.pop("annotations", None)
        dataset_dict.pop("sem_seg_file_name", None)
        
        image_size = dataset_dict['instances'].image_size
        dataset_dict['instances'] = Instances(
            image_size=image_size,
            gt_boxes=Boxes([]),
            gt_classes=torch.tensor([], dtype=torch.int64)
        )
        return dataset_dict
#----------End Define Labeled / Unlabeled Dataset Mapper----------#

#----------Start Define Semi-supervised Dataset Loader------------#
class Dualloaders:        
    def __init__(self, loader1, loader2):
        self.loader1 = iter(loader1) if loader1 is not None else itertools.repeat(None)
        self.loader2 = iter(loader2) if loader2 is not None else itertools.repeat(None)
    
    def __iter__(self):
        while True:
            yield (next(self.loader1), next(self.loader2))

class SemiDataloader:
    def __init__(self, labeled_loader, unlabeled_loader, batch_contents=("labeled_weak", "labeled_strong", "unlabeled_strong")):
        self.loader = Dualloaders(labeled_loader, unlabeled_loader)
        self.batch_contents = batch_contents
    
    def __iter__(self):
        for batch in self.loader:
            yield unpack_semi_data(*batch, batch_contents=self.batch_contents)

    def __len__(self):
        return len(self.loader)

def unpack_semi_data(labeled, unlabeled, batch_contents=("labeled_weak", "labeled_strong", "unlabeled_strong")):
    labeled_weak = None
    if "labeled_weak" in batch_contents and labeled is not None:
        labeled_weak = copy.deepcopy(labeled)
        for img in labeled_weak:
            if WEAK_IMG_KEY in img:
                img["image"] = img[WEAK_IMG_KEY]
    labeled_strong = labeled if "labeled_strong" in batch_contents else None

    unlabeled_weak = None
    if ("unlabeled_weak" in batch_contents or "unlabeled_strong" in batch_contents) and unlabeled is not None:
        unlabeled_weak = copy.deepcopy(unlabeled)
        for img in unlabeled_weak:
            if WEAK_IMG_KEY in img:
                img["image"] = img[WEAK_IMG_KEY]
    unlabeled_strong = unlabeled if "unlabeled_strong" in batch_contents else None

    return labeled_weak, labeled_strong, unlabeled_weak, unlabeled_strong
