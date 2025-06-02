import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from detectron2.modeling.sampling import subsample_labels
from detectron2.modeling.box_regression import _dense_box_regression_loss
from detectron2.utils.registry import Registry
from detectron2.config import configurable
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.layers import cat
from detectron2.layers.wrappers import cross_entropy
from fvcore.nn import smooth_l1_loss

from detectron2.structures.boxes import Boxes
from detectron2.structures.instances import Instances

from da2od.utils import SaveIO, RandomSeed, set_attributes

DISTILLER_REGISTRY = Registry("DISTILLER")

def build_distiller(cfg, teacher, student):
    name = cfg.DA.DISTILL.DISTILLER_NAME
    return DISTILLER_REGISTRY.get(name).from_config(cfg, teacher, student)

@DISTILLER_REGISTRY.register()
class Distiller:
    def __init__(self, teacher, student):
        pass

    @classmethod
    def from_config(cls, cfg, teacher, student):
        return Distiller(teacher, student)

    def __call__(self, teacher_batched_inputs, student_batched_inputs):
        return {}

    def is_distill_enabled(self):
        return False

class DistillMixin(GeneralizedRCNN): pass

@DISTILLER_REGISTRY.register()
class DA2ODDistiller(Distiller):
    """Knowledge Distillation that return prediction divergence for Faster R-CNN models."""
    def __init__(self, teacher, student, 
                 do_hard_cls=False, do_hard_obj=False, do_hard_rpn_reg=False, do_hard_roi_reg=False,
                 do_soft_cls=False, do_soft_obj=False, do_soft_rpn_reg=False, do_soft_roih_reg=False,
                 cls_temperature=1.0, obj_temperature=1.0, cls_loss_type="CE", pseudo_label_threshold=0.8):
        set_attributes(self, locals())
        self.register_hooks()
        self.pseudo_labeler = PseudoLabeler(teacher, pseudo_label_threshold)

    @classmethod
    def from_config(cls, cfg, teacher, student):
        return DA2ODDistiller(teacher, student,
                        do_hard_cls=cfg.DA.DISTILL.HARD_ROIH_CLS_ENABLED,
                        do_hard_obj=cfg.DA.DISTILL.HARD_OBJ_ENABLED,
                        do_hard_rpn_reg=cfg.DA.DISTILL.HARD_RPN_REG_ENABLED,
                        do_hard_roi_reg=cfg.DA.DISTILL.HARD_ROIH_REG_ENABLED,
                        do_soft_cls=cfg.DA.DISTILL.SOFT_ROIH_CLS_ENABLED, 
                        do_soft_obj=cfg.DA.DISTILL.SOFT_OBJ_ENABLED,
                        do_soft_rpn_reg=cfg.DA.DISTILL.SOFT_RPN_REG_ENABLED,
                        do_soft_roih_reg=cfg.DA.DISTILL.SOFT_ROIH_REG_ENABLED,
                        cls_temperature=cfg.DA.DISTILL.CLS_TMP,
                        obj_temperature=cfg.DA.DISTILL.OBJ_TMP,
                        cls_loss_type=cfg.DA.CLS_LOSS_TYPE,
                        pseudo_label_threshold=cfg.DA.TEACHER.THRESHOLD)

    def register_hooks(self):
        self.student_hooks = {
            "rpn_io": SaveIO(),
            "rpn_head_io": SaveIO(),
            "boxpred_io": SaveIO(),
            "boxhead_io": SaveIO()
        }
        self.teacher_hooks = {
            "backbone_io": SaveIO(),
            "rpn_head_io": SaveIO(),
            "boxpred_io": SaveIO(),
            "anchor_io": SaveIO(),
            "boxhead_io": SaveIO()
        }

        student_model = self.student.module if isinstance(self.student, DDP) else self.student
        teacher_model = self.teacher.module if isinstance(self.teacher, DDP) else self.teacher

        student_model.proposal_generator.register_forward_hook(self.student_hooks["rpn_io"])
        student_model.proposal_generator.rpn_head.register_forward_hook(self.student_hooks["rpn_head_io"])
        student_model.roi_heads.box_predictor.register_forward_hook(self.student_hooks["boxpred_io"])

        teacher_model.proposal_generator.anchor_generator.register_forward_hook(self.teacher_hooks["anchor_io"])
        teacher_model.proposal_generator.rpn_head.register_forward_hook(self.teacher_hooks["rpn_head_io"])
        teacher_model.roi_heads.box_predictor.register_forward_hook(self.teacher_hooks["boxpred_io"])
        
        self.seed = RandomSeed()
        teacher_model.roi_heads.register_forward_pre_hook(self.seed)
        student_model.roi_heads.register_forward_pre_hook(self.seed)

        self.teacher_proposal_replacer = ReplaceProposals()
        teacher_model.roi_heads.register_forward_pre_hook(self.teacher_proposal_replacer)

    def is_distill_enabled(self):
        return any([
            self.do_hard_cls, self.do_hard_obj, self.do_hard_rpn_reg, self.do_hard_roi_reg,
            self.do_soft_cls, self.do_soft_obj, self.do_soft_rpn_reg, self.do_soft_roih_reg
        ])

    def pseudo_labeling_and_proposal_replacing(self, teacher_inputs, student_inputs):
        """
        Adding the pseudo labels to the student inputs
        Do a student forwording process, and get the loss
        Replacing the teacher's proposals with student's.
        """
        self.pseudo_labeler(teacher_inputs, student_inputs)
        self.seeder.reset_seed()

        is_eval = not self.teacher.training
        if is_eval:
            self.teacher.train()

        standard_losses = self.student(student_inputs)  # use the pseudo labels from teacher calculating loss
        student_proposals, _ = self.student_hooks["rpn_io"].output

        self.teacher_proposal_replacer.set_proposals(student_proposals)
        with torch.no_grad():
            self.teacher(teacher_inputs)

        if is_eval:
            self.teacher.eval()

        return standard_losses

    def __call__(self, teacher_inputs, student_inputs):
        hard_losses = self.pseudo_labeling_and_proposal_replacing(teacher_inputs, student_inputs)

        whether_to_add_loss = {
            "loss_cls": self.do_hard_cls,
            "loss_rpn_cls": self.do_hard_obj,
            "loss_rpn_loc": self.do_hard_rpn_reg,
            "loss_box_reg": self.do_hard_roi_reg,
        }

        losses = {
            k: v if whether_to_add_loss.get(k, False) else v * 0.0
            for k, v in hard_losses.items()
        }
        
        losses.update(self.compute_soft_rpn_losses(teacher_inputs))
        losses.update(self.compute_soft_roih_losses())

        # calculate the prediction divergence between teacher and student
        student_cls_logits, _ = self.student_hooks["boxpred_io"].output
        teacher_cls_logits, _ = self.teacher_hooks["boxpred_io"].output

        diff = (student_cls_logits - teacher_cls_logits) ** 2
        diff = torch.mean(diff, dim=1, keepdim=True)

        pseudo_boxes = [input['instances'].gt_boxes.tensor.cuda() for input in student_inputs]

        return losses, diff, pseudo_boxes[:]

    def compute_soft_rpn_losses(self, teacher_inputs):
        losses = {}
        student_objectness, student_deltas = self.student_hooks["rpn_head_io"].output
        teacher_objectness, teacher_deltas = self.teacher_hooks["rpn_head_io"].output

        rpn = self.teacher.module.proposal_generator if isinstance(self.teacher, DDP) else self.teacher.proposal_generator
        pseudo_gt_labels = torch.stack(rpn.label_and_sample_anchors(
            self.teacher_hooks["anchor_io"].output,
            [i['instances'].to(self.teacher.device) for i in teacher_inputs]
        )[0])

        valid_mask = torch.flatten(pseudo_gt_labels >= 0)
        fg_mask = pseudo_gt_labels == 1

        teacher_objectness_probs = torch.sigmoid(cat([torch.flatten(t) for t in teacher_objectness]) / self.obj_temperature)

        if self.do_soft_obj:
            objectness_loss = F.binary_cross_entropy_with_logits(
                cat([torch.flatten(t) for t in student_objectness])[valid_mask],
                teacher_objectness_probs[valid_mask],
                reduction="mean"
            )
            losses["loss_obj_bce"] = objectness_loss

        if self.do_soft_rpn_reg:
            fg_mask = torch.repeat_interleave(fg_mask, repeats=4)
            loss_rpn_reg = smooth_l1_loss(
                cat([torch.flatten(t) for t in student_deltas])[fg_mask],
                cat([torch.flatten(t) for t in teacher_deltas])[fg_mask],
                beta=0.0,
                reduction="mean"
            )
            losses["loss_rpn_l1"] = loss_rpn_reg

        return losses

    def compute_soft_roih_losses(self):
        losses = {}
        student_cls, student_deltas = self.student_hooks["boxpred_io"].output
        teacher_cls, teacher_deltas = self.teacher_hooks["boxpred_io"].output

        teacher_probs = F.softmax(teacher_cls / self.cls_temperature, dim=1)

        if self.do_soft_obj:
            if self.cls_loss_type == "CE":
                cls_loss = cross_entropy(student_cls, teacher_probs)
            elif self.cls_loss_type == "KL":
                cls_loss = F.kl_div(
                    F.log_softmax(student_cls, dim=1),
                    F.log_softmax(teacher_cls / self.cls_temperature, dim=1),
                    reduction="batchmean",
                    log_target=True
                )
            else:
                raise ValueError("cls_loss_type must be 'CE' or 'KL'")
            losses["loss_cls_ce"] = cls_loss

        if self.do_soft_roih_reg:
            bg_idx = teacher_cls.shape[1] - 1
            fg_cls = torch.argmax(teacher_cls, dim=1)
            fg_mask = fg_cls != bg_idx

            fg_teacher_deltas = teacher_deltas.view(-1, bg_idx, 4)[fg_mask, fg_cls[fg_mask], :]
            fg_student_deltas = student_deltas.view(-1, bg_idx, 4)[fg_mask, fg_cls[fg_mask], :]

            loss_roih_reg = smooth_l1_loss(
                fg_student_deltas,
                fg_teacher_deltas,
                beta=0.0,
                reduction="sum"
            )

            normalizer = teacher_cls.shape[0]
            losses["loss_roih_l1"] = loss_roih_reg / normalizer

        return losses
    
class ReplaceProposals:
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

class PseudoLabeler:
    def __init__(self, model, threshold):
        self.model = model
        self.threshold = threshold

    def __call__(self, unlabeled_weak, unlabeled_strong):
        with torch.no_grad():
            was_training = self.model.training
            self.model.eval()
            teacher_preds = self.model.inference(unlabeled_weak, do_postprocess=False)
            if was_training: self.model.train()

            teacher_preds = self.threshold_pseudo_label(teacher_preds, self.threshold)
            
            self.add_label(unlabeled_weak, teacher_preds)
            if unlabeled_strong is not None:
                self.add_label(unlabeled_strong, teacher_preds)
    
    # Modified from Adaptive Teacher
    def threshold_pseudo_label(self, proposals, cur_threshold):
        list_instances = []
        for proposal_bbox_inst in proposals:
            proposal_bbox_inst = self.process_bbox(
                proposal_bbox_inst,
                thres=cur_threshold, 
            )
            list_instances.append(proposal_bbox_inst)
            
        return list_instances

    # Modified from Adaptive Teacher
    def process_bbox(self, proposal_bbox_inst, thres=0.7):
        valid_map = proposal_bbox_inst.scores > thres

        # create instances containing boxes and gt_classes
        image_shape = proposal_bbox_inst.image_size
        new_proposal_inst = Instances(image_shape)

        # create box
        new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor[valid_map, :]
        new_boxes = Boxes(new_bbox_loc)

        # add boxes to instances
        new_proposal_inst.gt_boxes = new_boxes.to("cpu")
        new_proposal_inst.gt_classes = proposal_bbox_inst.pred_classes[valid_map].to("cpu")
        new_proposal_inst.scores = proposal_bbox_inst.scores[valid_map].to("cpu")

        return new_proposal_inst

    def add_label(self, unlabled_data, label):
        for unlabel_datum, lab_inst in zip(unlabled_data, label):
            unlabel_datum["instances"] = lab_inst
        return unlabled_data