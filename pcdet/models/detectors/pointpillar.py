from .detector3d_template import Detector3DTemplate
from .unet.unet import UNet, SimplifiedUNet
from .unet.resunet import ResUnet_64, ResUnet_128 


from .segmentation_head import FCNMaskHead
import sys
from .erfnet import Net
import os
import torch.nn.functional as F
import torch.nn as nn
import torch
from ...ops.roiaware_pool3d import roiaware_pool3d_utils

from .. import backbones_image, view_transforms


from PIL import Image
import numpy as np
from torch.autograd import Variable
try:
    from itertools import  ifilterfalse
except ImportError: # py3k
    from itertools import  filterfalse as ifilterfalse


def isnan(x):
    return x != x

def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n

def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    if per_image:
        loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
                          for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), classes=classes)
    return loss


def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float() # foreground for class c
        if (classes is 'present' and fg.sum() == 0):
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (Variable(fg) - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)


def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    if probas.dim() == 3:
        # assumes output of a sigmoid layer
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    elif probas.dim() == 5:
        #3D segmentation
        B, C, L, H, W = probas.size()
        probas = probas.contiguous().view(B, C, L, H*W)
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels

    
class FocalLoss(nn.Module):
    """
    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, apply_nonlin=None, alpha=None, gamma=2, balance_index=0, smooth=1e-5, size_average=True):
        super(FocalLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, logit, target):
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size()[0], logit.size()[1], logit.size()[3])
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = torch.unsqueeze(target, 1)
        target = target.view(-1, 1)
        # print(logit.shape, target.shape)
        # 
        alpha = self.alpha

        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha

        else:
            raise TypeError('Not support alpha type')

        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size()[0], num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth / (num_class - 1), 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


def one_hot(labels, num_classes):
    '''
    Converts an integer label torch.autograd.Variable to a one-hot Variable.

    Parameters
    ----------
    labels : torch.autograd.Variable of torch.cuda.LongTensor
        N x 1 x H x W, where N is batch size.
        Each value is an integer representing correct classification.
    Returns
    -------
    target : torch.autograd.Variable of torch.cuda.FloatTensor
        N x C x H x W, where C is class number. One-hot encoded.
    '''
    one_hot = torch.cuda.FloatTensor(labels.size()[0], num_classes, labels.size()[2], labels.size()[3]).zero_()
    target = one_hot.scatter_(1, labels.data, 1)
    return target


def one_hot_1d(data, num_classes):
    n_values = num_classes
    n_values = torch.eye(n_values)[data]
    return n_values

def get_class_weights():
    '''
    Cless weights being 1/log(fc) (https://arxiv.org/pdf/2008.10559.pdf)
    freq: 一个包含各类别频率（或数量）的 NumPy 数组或列表
    '''

    class_frequencies = [6039821678, 75773262, 338027, 1216738, 41160, 1180111808, 685755462, 
                        93861877, 173027635, 70317464,716695190, 5906686, 522133013] # kitti 12类
    # class_frequencies = [5942257919, 5203822, 94890, 431723, 5106113, 952859, 525118, 686825, 1589979, 
    #                     675611908, 14480495, 162538038, 184926572, 171146559, 208557900] # nuscenes 14类

    freq = class_frequencies[1:]
    epsilon_w = 0.001  # eps to avoid zero division
    # 统一转为 numpy array，避免 list + float 错误
    freq = np.asarray(freq, dtype=np.float32)
    weights = torch.from_numpy(1 / np.log(freq + epsilon_w))

    return weights * 50

def calcu_wce_logits(probas, labels, num_class=12):
    '''
    probas: [1, 12, 500, 1000] -> [1, 12, point_num]
    labels: [1, 1, 500, 1000] -> [point_num] -> targets_crr[1, 12, point_num]
    '''
    nozero_mask = labels != 0
    labels = torch.clamp(labels[nozero_mask], 1, num_class) 
    labels = one_hot_1d((labels - 1).long(), num_class).unsqueeze(0).permute(0, 2, 1).contiguous().cuda() 
    # labels[1, 1, 500, 1000] -> targets_crr_1d[124482] -> targets_crr[1, 12, 124482]
    probas = probas.permute(0, 2, 3, 1).unsqueeze(1)[nozero_mask].squeeze().unsqueeze(0).permute(0, 2, 1).contiguous()
    # pred: [1, 12, 500, 1000]-> [1, 12, 124482]

    # 手动调整kitti权重 -------------------------------------------------------------
    # for dense
    # weight = torch.ones_like(labels)
    # weight[:, 0, :] = 2  
    # weight[:, [1, 2, 3], :] = 7.5  
    # # for sparse
    # weight[:, 0, :] = 2  # weight 5 for vehicle
    # weight[:, [1, 2, 3], :] = 8  # weight8 for person, two wheel and rider
    # ------------------------------------------------------------------------------

    # # 根据数据集选择权重 -------------------------------------------------------------
    kitti_weights = get_class_weights().to(labels.device).float() # 用于分割任务的类别权重 12、14类
    weight = kitti_weights.view(1, num_class, 1).expand_as(labels)
    # # ------------------------------------------------------------------------------

    # 手动调整nuscenes权重 -------------------------------------------------------------
    # for dense
    # weight = torch.ones_like(labels)
    # weight[:, 0, :] = 2  
    # weight[:, [1, 5, 6], :] = 7.5  
    # # for sparse
    # weight[:, 0, :] = 2  # weight 5 for vehicle
    # weight[:, [1, 2, 3], :] = 8  # weight8 for person, two wheel and rider
    # ------------------------------------------------------------------------------

    loss_wce = F.binary_cross_entropy_with_logits(probas, labels, reduction='mean', weight=weight)

    return loss_wce

def calcu_lovasz_softmax(probas, labels):
    
    '''
    probas: [1, 12, 500, 1000] -> [B, C, H, W] 
    labels: [1, 1, 500, 1000] -> [B, H, W] 
    probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
    labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
    '''
    labels = labels.squeeze(0)
    loss_lovasz = lovasz_softmax(probas, labels)

    return loss_lovasz


class PointPillar(Detector3DTemplate):
    """
    MODEL:
    NAME: PointPillar

    VFE:
        NAME: PillarVFE
        WITH_DISTANCE: False
        USE_ABSLOTE_XYZ: True
        USE_NORM: True
        NUM_FILTERS: [64]
    """

    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        # self.module_topology = [
        #     'vfe', 'map_to_bev_module', 
        #     'image_backbone', 'vtransform',
        #     'backbone_2d', 'dense_head'
        # ]
        self.module_list = self.build_networks()   # {PillarVEF, pointPillarScatter}

        self.segmentation_head = ResUnet_128(128, 12)

        self.focal_loss = FocalLoss()

    def build_image_backbone(self, model_info_dict):
        # SwinT_tiny_Encoder带了FPN，所以不需要再构建FPN
        if self.model_cfg.get('IMAGE_BACKBONE', None) is None:
            return None, model_info_dict
        image_backbone_module = backbones_image.__all__[self.model_cfg.IMAGE_BACKBONE.NAME](
            model_cfg=self.model_cfg.IMAGE_BACKBONE
        )
        # image_backbone_module.init_weights()
        model_info_dict['module_list'].append(image_backbone_module)

        return image_backbone_module, model_info_dict

    def build_vtransform(self,model_info_dict):
        if self.model_cfg.get('VTRANSFORM', None) is None:
            return None, model_info_dict
        
        vtransform_module = view_transforms.__all__[self.model_cfg.VTRANSFORM.NAME](
            model_cfg=self.model_cfg.VTRANSFORM
        )
        model_info_dict['module_list'].append(vtransform_module)

        return vtransform_module, model_info_dict


    def forward(self, batch_dict):
        # print("[DEBUG] Module List:", [m.__class__.__name__ for m in self.module_list])
        module_index = 0

        for cur_module in self.module_list[:2]: # 2,4
            module_index += 1
            batch_dict = cur_module(batch_dict)
            
            if module_index == 2: # 2
                # print("[DEBUG] cur: ", cur_module.__class__.__name__, "module_index: ", module_index)

                dict_seg = []
                dict_cls_num = []
                label_b = batch_dict["labels_seg"]

                # batch, c, h, w = label_b.size()
                # targets_crr = label_b.view(batch, c, h, w)  # torch.cat(dict_seg,dim=0).view(batch,c,h,w)
                targets_crr = label_b
                lidar_features = batch_dict["spatial_features"]

                pred = self.segmentation_head(lidar_features)#lidar_features, image_features) fused_features

                batch_dict['prediction'] = pred
                batch_dict['fullprediction'] = pred

        """
           code for geomertic consistency
        """
        if self.training:
            loss_lovasz = lovasz_softmax(torch.nn.functional.softmax(pred, dim=-1), targets_crr)
            loss_wce = calcu_wce_logits(pred, targets_crr, num_class=12)
            loss_seg = loss_wce * 1.0 + loss_lovasz * 0.0
            
            ret_dict = {
                'loss': loss_seg
            }
            disp_dict = {}
            tb_dict = {}
            return ret_dict, tb_dict, disp_dict  # 对应loss, tb_dict, disp_dict = model_func(model, batch)
        else:
            return batch_dict # 对应pred_dict = model(batch_dict)
            # pred_dicts, recall_dicts = self.post_processing(batch_dict)
            # return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict
