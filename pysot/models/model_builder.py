# Copyright (c) SenseTime. All Rights Reserved.
####@author：ZZXin
####@data：2019/7/10/
####@modify：添加经过backbone后zf/xf可视化

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.nn as nn
import torch.nn.functional as F
import torch
import cv2
import numpy as np

from pysot.core.config import cfg
from pysot.models.loss import select_cross_entropy_loss, weight_l1_loss
from pysot.models.backbone import get_backbone
from pysot.models.head import get_rpn_head, get_mask_head, get_refine_head
from pysot.models.neck import get_neck
from pysot.tracker.siamrpn_tracker import SiamRPNTracker as tracker

#### TADT
from pysot.tadt.taf import taf_model
from pysot.tadt.feature_utils_v2 import features_selection


class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()

        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)

        # build adjust layer
        if cfg.ADJUST.ADJUST:
            self.neck = get_neck(cfg.ADJUST.TYPE,
                                 **cfg.ADJUST.KWARGS)

        # build rpn head
        self.rpn_head = get_rpn_head(cfg.RPN.TYPE,
                                     **cfg.RPN.KWARGS)

        # build mask head
        if cfg.MASK.MASK:
            self.mask_head = get_mask_head(cfg.MASK.TYPE,
                                           **cfg.MASK.KWARGS)

            if cfg.REFINE.REFINE:
                self.refine_head = get_refine_head(cfg.REFINE.TYPE)
    '''
    def neck_tadt(self,zf,frame_id = None):
        zf = self.neck(zf)
        self.zf = zf
    '''
    # 用于从提取得到的x_crop的featuremap来crop得到z_crop
    def generate_patch_feature(self, features, exemplar_size, instance_size):
        patch_features = []
        patch_locations = []

        for feature in features:
            feature_size = torch.tensor(feature.shape[-2:]).numpy()

            center = np.floor(feature_size / 2) + 1

            patch_size = np.floor(exemplar_size * feature_size / instance_size / 2) * 2 + 1

            patch_loc = np.append(center - np.floor(patch_size / 2), center + np.floor(patch_size / 2)).astype(int)

            patch_feature = feature[:, :, patch_loc[0]:patch_loc[2] + 1, patch_loc[1]:patch_loc[3] + 1]
            patch_features.append(patch_feature)
            patch_locations.append(patch_loc)

        return patch_features, patch_locations

    def template(self, x, z, frame_id=None):
        xf = self.backbone(x)

        self.xf = xf  # 保证后面的一致性
        # 再在x_crop的featuremap的基础上crop得到z_crop的特征图 - 论文中有对这两种特征图的获取方式作对比实验分析
        # zf, zf_location = self.generate_patch_feature(self.xf, [cfg.TRACK.EXEMPLAR_SIZE, cfg.TRACK.EXEMPLAR_SIZE], [cfg.TRACK.INSTANCE_SIZE, cfg.TRACK.INSTANCE_SIZE])
        # zf_nouse, zf_location = self.generate_patch_feature(self.xf, [cfg.TRACK.EXEMPLAR_SIZE, cfg.TRACK.EXEMPLAR_SIZE], [cfg.TRACK.INSTANCE_SIZE, cfg.TRACK.INSTANCE_SIZE])
        zf = self.backbone(z)      # 这一行和上一行注释掉,做模板帧特征图获取方式结果对比-2019/10/23-

        # 添加可视化-可视化经过backbone提取到的特征zf
        visualize = False # 可视化标志位
        if visualize:
            for i in range(3):
                xfi = self.xf[i] # 可视化第1帧的search area的feature map
                heatmap = torch.sum(xfi, dim=1)
                # 作归一化 处理
                max_value = torch.max(heatmap)
                min_value = torch.min(heatmap)
                heatmap = (heatmap - min_value) / (max_value - min_value) * 255
                heatmap = heatmap.cpu().detach().numpy().astype(np.uint8).transpose(1, 2, 0)
                heatmap = cv2.resize(heatmap, (255, 255), interpolation=cv2.INTER_LINEAR)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX,
                              cv2.CV_8U)
                str0 = str(frame_id) + "_" + "x_crop" + "_" + "layer" + str(i + 2)  # 图片名称
                str1 = "/media/db/dbVolume/Datasets/pysot/images/images_paper/" + str0 + ".jpg"  # 图片保存路径
                cv2.imwrite(str1, heatmap)  # 保存图片
                # cv2.imshow(str0, heatmap)
                # cv2.waitKey(1000)

            for i in range(3):
                xfi = zf[i]  # 可视化第1帧的template的feature map
                heatmap = torch.sum(xfi, dim=1)
                # 作归一化 处理
                max_value = torch.max(heatmap)
                min_value = torch.min(heatmap)
                heatmap = (heatmap - min_value) / (max_value - min_value) * 255
                heatmap = heatmap.cpu().detach().numpy().astype(np.uint8).transpose(1, 2, 0)
                heatmap = cv2.resize(heatmap, (127, 127), interpolation=cv2.INTER_LINEAR)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX,
                              cv2.CV_8U)
                str0 = str(frame_id) + "_" + "z_crop" + "_" + "layer" + str(i + 2)  # 图片名称
                str1 = "/media/db/dbVolume/Datasets/pysot/images/images/" + str0 + ".jpg"  # 图片保存路径
                cv2.imwrite(str1, heatmap)  # 保存图片
                # cv2.imshow(str0, heatmap)
                # cv2.waitKey(1000)

        #### TADT部分
        self.filter_sizes = [torch.tensor(feature.shape).numpy() for feature in zf]  # 由target的特征图大小计算得到相应的filter大小
        # self.feature_weights保存了各个层的提取的通道信息-要提取的channel置1
        self.feature_weights = taf_model(self.xf, self.filter_sizes, 'cuda')
        self.exemplar_features = features_selection(zf, self.feature_weights, mode='reduction')
        zf = self.exemplar_features

        visualize = False
        if visualize:
            for i in range(3):
                xfi = zf[i] # 可视化第1帧的提取特征之后的exemplar_features的feature map
                heatmap = torch.sum(xfi, dim=1)
                # 作归一化 处理
                max_value = torch.max(heatmap)
                min_value = torch.min(heatmap)
                heatmap = (heatmap - min_value) / (max_value - min_value) * 255
                heatmap = heatmap.cpu().detach().numpy().astype(np.uint8).transpose(1, 2, 0)
                heatmap = cv2.resize(heatmap, (127, 127), interpolation=cv2.INTER_LINEAR)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX,
                              cv2.CV_8U)
                str0 = str(frame_id) + "_" + "z_crop_selected" + "_" + "layer" + str(i + 2)  # 图片名称
                str1 = "/media/db/dbVolume/Datasets/pysot/images/images/" + str0 + ".jpg"  # 图片保存路径
                cv2.imwrite(str1, heatmap)  # 保存图片
                # cv2.imshow(str0, heatmap)
                # cv2.waitKey(1000)

        if cfg.MASK.MASK:
            zf = zf[-1]

        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)

        self.zf = zf

    def track(self, x,frame_id = None):
        xf = self.backbone(x)

        #### 添加可视化-可视化经过backbone提取到的特征xf
        visualize = False  # 可视化标志位
        if visualize:
            for i in range(3):
                xfi = xf[i]
                heatmap = torch.sum(xfi, dim=1)
                # 作归一化 处理
                max_value = torch.max(heatmap)
                min_value = torch.min(heatmap)
                heatmap = (heatmap - min_value) / (max_value - min_value) * 255
                heatmap = heatmap.cpu().detach().numpy().astype(np.uint8).transpose(1, 2, 0)
                heatmap = cv2.resize(heatmap, (255, 255), interpolation=cv2.INTER_LINEAR)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX,
                              cv2.CV_8U)  # 将response_map的图像格式从float32转成uint8-解决保存图片全黑的问题
                str0 = str(frame_id) + "_" + "x_crop" + "_" + "layer" + str(i + 2)  # 图片名称
                str1 = "/media/db/dbVolume/Datasets/pysot/images/images/" + str0 + ".jpg"  # 图片保存路径
                cv2.imwrite(str1, heatmap)  # 保存图片
                ####?cv2.imshow(str0, heatmap)
                ####?cv2.waitKey(1000)

        # 提取对target activate的channel
        self.instance_features = features_selection(xf, self.feature_weights, mode='reduction')
        xf = self.instance_features

        visualize = False  # 可视化标志位
        if visualize:
            for i in range(3):
                xfi = xf[i] # 可视化经过特征筛选之后的feature map
                heatmap = torch.sum(xfi, dim=1)
                # 作归一化 处理
                max_value = torch.max(heatmap)
                min_value = torch.min(heatmap)
                heatmap = (heatmap - min_value) / (max_value - min_value) * 255
                heatmap = heatmap.cpu().detach().numpy().astype(np.uint8).transpose(1, 2, 0)
                heatmap = cv2.resize(heatmap, (255, 255), interpolation=cv2.INTER_LINEAR)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX,
                              cv2.CV_8U)
                str0 = str(frame_id) + "_" + "x_crop_selected" + "_" + "layer" + str(i + 2)  # 图片名称
                str1 = "/media/db/dbVolume/Datasets/pysot/images/images/" + str0 + ".jpg"  # 图片保存路径
                cv2.imwrite(str1, heatmap)  # 保存图片
                # cv2.imshow(str0, heatmap)
                # cv2.waitKey(1000)

        if cfg.MASK.MASK:
            self.xf = xf[:-1]
            xf = xf[-1]
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)

        cls, loc = self.rpn_head(self.zf, xf,frame_id = frame_id)

        if cfg.MASK.MASK:
            mask, self.mask_corr_feature = self.mask_head(self.zf, xf)
        return {
                'cls': cls,
                'loc': loc,
                'mask': mask if cfg.MASK.MASK else None
               }

    def mask_refine(self, pos):
        return self.refine_head(self.xf, self.mask_corr_feature, pos)

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2//2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls

    # 模型的前向传播的核心函数-注意传进来的data数据
    def forward(self, data):
        """ only used in training
        """
        template = data['template'].cuda()
        search = data['search'].cuda()
        label_cls = data['label_cls'].cuda()
        label_loc = data['label_loc'].cuda()
        label_loc_weight = data['label_loc_weight'].cuda()

        # get feature
        zf = self.backbone(template)
        xf = self.backbone(search)

        if cfg.MASK.MASK:
            zf = zf[-1]
            self.xf_refine = xf[:-1]
            xf = xf[-1]
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
            xf = self.neck(xf)
        cls, loc = self.rpn_head(zf, xf)

        # get loss
        cls = self.log_softmax(cls)
        cls_loss = select_cross_entropy_loss(cls, label_cls)
        loc_loss = weight_l1_loss(loc, label_loc, label_loc_weight)

        outputs = {}
        outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
            cfg.TRAIN.LOC_WEIGHT * loc_loss
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss

        if cfg.MASK.MASK:
            # TODO
            mask, self.mask_corr_feature = self.mask_head(zf, xf)
            mask_loss = None
            outputs['total_loss'] += cfg.TRAIN.MASK_WEIGHT * mask_loss
            outputs['mask_loss'] = mask_loss
        return outputs
