import torch
import torch.nn.functional as F

from detectron2.config import configurable
from detectron2.modeling import GeneralizedRCNN

from da2od.utils import SaveIO, grad_reverse


class AlignMixin(GeneralizedRCNN):
    @configurable
    def __init__(
        self,
        *,
        img_da_enabled: bool = False,
        img_da_layer: str = None,
        img_da_weight: float = 0.0,
        img_da_input_dim: int = 256,
        img_da_hidden_dims: list = [256,],
        ins_da_enabled: bool = False,
        ins_da_weight: float = 0.0,
        ins_da_input_dim: int = 1024,
        ins_da_hidden_dims: list = [1024,],
        **kwargs
    ):
        super(AlignMixin, self).__init__(**kwargs)
        self.img_da_layer = img_da_layer
        self.img_da_weight = img_da_weight
        self.ins_da_weight = ins_da_weight

        self.img_align = ConvDiscriminator(img_da_input_dim, hidden_dims=img_da_hidden_dims) if img_da_enabled else None
        self.ins_align = FCDiscriminator(ins_da_input_dim, hidden_dims=ins_da_hidden_dims) if ins_da_enabled else None 
        # self.foreback_weight = torch.nn.Parameter(torch.tensor(0.5))
        self.epsilon = 0.1
        self.fore_weight = 0.8

        # register hooks so we can grab output of sub-modules
        self.backbone_io, self.rpn_io, self.roih_io, self.boxhead_io = SaveIO(), SaveIO(), SaveIO(), SaveIO()
        self.backbone.register_forward_hook(self.backbone_io)
        self.proposal_generator.register_forward_hook(self.rpn_io)
        self.roi_heads.register_forward_hook(self.roih_io)       

        if ins_da_enabled:
            assert hasattr(self.roi_heads, 'box_head'), "Instance alignment only implemented for ROI Heads with box_head."
            self.roi_heads.box_head.register_forward_hook(self.boxhead_io)

    @classmethod
    def from_config(cls, cfg):
        ret = super(AlignMixin, cls).from_config(cfg)

        ret.update({"img_da_enabled": cfg.DA.ALIGN.IMG_DA_ENABLED,
                    "img_da_layer": cfg.DA.ALIGN.IMG_DA_LAYER,
                    "img_da_weight": cfg.DA.ALIGN.IMG_DA_WEIGHT,
                    "img_da_input_dim": cfg.DA.ALIGN.IMG_DA_INPUT_DIM,
                    "img_da_hidden_dims": cfg.DA.ALIGN.IMG_DA_HIDDEN_DIMS,
                    "ins_da_enabled": cfg.DA.ALIGN.INS_DA_ENABLED,
                    "ins_da_weight": cfg.DA.ALIGN.INS_DA_WEIGHT,
                    "ins_da_input_dim": cfg.DA.ALIGN.INS_DA_INPUT_DIM,
                    "ins_da_hidden_dims": cfg.DA.ALIGN.INS_DA_HIDDEN_DIMS,
                    })

        return ret
    
    # boxes is list containing 2 tensor, len(boxes) = 2, based on height and width
    def box_to_mask(self, bs_boxes, size_feat, size_ori_img):
        mask = torch.zeros(size_feat).cuda()
        if bs_boxes == None:
            return mask
        feat_w, feat_h = size_feat[-1], size_feat[-2]
        img_w, img_h = size_ori_img[-1], size_ori_img[-2]
        scale_x, scale_y = torch.tensor([feat_w/img_w]), torch.tensor([feat_h/img_h])
        scale_fct = torch.stack([scale_x, scale_y, scale_x, scale_y]).view(1,4).cuda()
        
        for i, bs_box in enumerate(bs_boxes):
            for box in bs_box:
                if len(box.size()) != 3:
                    box = box.unsqueeze(0)
                    box = box * scale_fct
                    box = box.squeeze(0)
                    xmin, ymin, xmax, ymax = box
                    xmin, xmax, ymin, ymax = int(xmin), int(xmax), int(ymin), int(ymax)
                    mask[i][:, ymin:ymax, xmin:xmax] = 1
            # mask[i] = mask[i]*(1-epsilon) + 0.5*epsilon
        return mask

    def forward(self, *args, do_align=False, alpha=1.0, pred_diff = None, pseudo_boxes = None, **kwargs):
        output = super().forward(*args, **kwargs)
        batch_inputs = args   # use for abtaining source image gt boxes
        src_gt_boxes = list()
        for inputs in batch_inputs:
            for sample in inputs:
                if 'instances' in sample:
                    src_gt_boxes.append(sample['instances'].gt_boxes.tensor.cuda())
            size_ori_img = [inputs[0]['height'], inputs[0]['width']]
        # for debug, bs=2
        # image_size = (960,1920), (832, 1664)
        # height: 1024, width: 2048
        # size = [2, 240, 480]
        if self.training:
            if do_align:
                # extract needed info for alignment: domain labels, image features, instance features
                # domain_label = 1 if labeled else 0
                domain_label = alpha
                img_features = list(self.backbone_io.output.values())
                device = img_features[0].device
                if self.img_align:
                    # features = self.backbone_io.output
                    # features = grad_reverse(features[self.img_da_layer])
                    # domain_preds = self.img_align(features)
                    # loss = F.binary_cross_entropy_with_logits(domain_preds, torch.FloatTensor(domain_preds.data.size()).fill_(domain_label).to(device))
                    # output["loss_da_img"] = self.img_da_weight * loss
                    features = self.backbone_io.output
                    # 在这里把用于预测domain-label的feature进行加权处理
                    size_feat = features[self.img_da_layer].shape
                    boxes = pseudo_boxes if pseudo_boxes is not None else src_gt_boxes
                    mask = self.box_to_mask(boxes, size_feat, size_ori_img)
                    mask_fore = mask*(1-self.epsilon) + 0.5*self.epsilon
                    mask_back  = (1-mask)*(1-self.epsilon) + 0.5*self.epsilon
                    feature_dis = features[self.img_da_layer]
                    
                    feature_fore = feature_dis * mask_fore   
                    features_reverse_fore = grad_reverse(feature_fore)
                    domain_preds_fore = self.img_align(features_reverse_fore)   
                    
                    feature_back = feature_dis * mask_back
                    features_reverse_back = grad_reverse(feature_back)
                    domain_preds_back = self.img_align(features_reverse_back)
                    
                    loss_fore = F.binary_cross_entropy_with_logits(domain_preds_fore, torch.FloatTensor(domain_preds_fore.data.size()).fill_(domain_label).to(device))
                    loss_back = F.binary_cross_entropy_with_logits(domain_preds_back, torch.FloatTensor(domain_preds_back.data.size()).fill_(domain_label).to(device))
                    
                    loss = self.fore_weight * loss_fore + (1.0-self.fore_weight) * loss_back
                    output["loss_da_img"] = self.img_da_weight * loss
                if self.ins_align:
                    instance_features = self.boxhead_io.output
                    features = grad_reverse(instance_features)
                    domain_preds = self.ins_align(features)
                    pred_diff = None
                    if pred_diff is None:
                        loss = F.binary_cross_entropy_with_logits(domain_preds, torch.FloatTensor(domain_preds.data.size()).fill_(domain_label).to(device))
                        output["loss_da_ins"] = self.ins_da_weight * loss
                    else:
                        # for debug
                        # print(instance_features.shape) torch.size([1024, 1024])
                        # print(features.shape) torch.size([1024, 1024])
                        # print(domain_preds.shape) torch.size([1024, 1])
                        if torch.isnan(pred_diff).any() or torch.isinf(pred_diff).any():
                            print("pred_diff contains NaN or Inf")
                            # 处理无效值，例如用0替换
                            pred_diff = torch.where(torch.isnan(pred_diff) | torch.isinf(pred_diff), torch.tensor(0.0, device="cuda"), pred_diff)
                        # 归一化
                        min_val = pred_diff.min()
                        max_val = pred_diff.max()
                        diff_weight = (pred_diff - min_val) / (max_val - min_val)
                        # diff_weight = F.softmax(pred_diff, dim=0)
                        # reverse the weights
                        ## change here
                        # diff_weight = torch.tensor(1.0, device='cuda') - diff_weight
                        diff_weight = diff_weight.detach()
                        loss = F.binary_cross_entropy_with_logits(domain_preds, torch.FloatTensor(domain_preds.data.size()).fill_(domain_label).to(device), weight=diff_weight, reduction='mean')
                        output["loss_da_ins"] = self.ins_da_weight * loss
            elif self.img_align or self.ins_align:
                # need to utilize the modules at some point during the forward pass or PyTorch complains.
                # this is only an issue when cfg.SOLVER.BACKWARD_AT_END=False, because intermediate backward()
                # calls may not have used alignment heads
                # see: https://github.com/pytorch/pytorch/issues/43259#issuecomment-964284292
                fake_output = 0
                for aligner in [self.img_align, self.ins_align]:
                    if aligner is not None:
                        fake_output += sum([p.sum() for p in aligner.parameters()]) * 0
                output["_da"] = fake_output
        return output

class ConvDiscriminator(torch.nn.Module):
    """A discriminator that uses conv layers."""
    def __init__(self, input_dim, hidden_dims=[], kernel_size=3):
        super(ConvDiscriminator, self).__init__()
        modules = []
        prev_dim = input_dim
        for i, dim in enumerate(hidden_dims):
            modules.append(torch.nn.Conv2d(prev_dim, dim, kernel_size))
            modules.append(torch.nn.ReLU())
            prev_dim = dim
        modules.append(torch.nn.AdaptiveAvgPool2d(1))
        modules.append(torch.nn.Flatten())
        # change here
        modules.append(torch.nn.Dropout(p=0.6))
        # change here
        modules.append(torch.nn.Linear(prev_dim, 1))
        
        self.model = torch.nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)
    
class FCDiscriminator(torch.nn.Module):
    """A discriminator that uses fully connected layers."""
    def __init__(self, input_dim, hidden_dims=[]):
        super(FCDiscriminator, self).__init__()
        modules = []
        modules.append(torch.nn.Flatten())
        prev_dim = input_dim
        for i, dim in enumerate(hidden_dims):
            modules.append(torch.nn.Linear(prev_dim, dim))
            modules.append(torch.nn.ReLU())
            prev_dim = dim
        modules.append(torch.nn.Linear(prev_dim, 1))
        self.model = torch.nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)