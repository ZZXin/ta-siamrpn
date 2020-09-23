# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import cv2
import torch

from pysot.core.config import cfg
from pysot.tracker.siamrpn_tracker import SiamRPNTracker


class SiamRPNLTTracker(SiamRPNTracker):
    def __init__(self, model):
        super(SiamRPNLTTracker, self).__init__(model)
        self.longterm_state = False

    def track(self, img, frame_id=None):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z

        if self.longterm_state:
            instance_size = cfg.TRACK.LOST_INSTANCE_SIZE
        else:
            instance_size = cfg.TRACK.INSTANCE_SIZE

        score_size = (instance_size - cfg.TRACK.EXEMPLAR_SIZE) // \
            cfg.ANCHOR.STRIDE + 1 + cfg.TRACK.BASE_SIZE
        hanning = np.hanning(score_size)
        window = np.outer(hanning, hanning)
        window = np.tile(window.flatten(), self.anchor_num)
        anchors = self.generate_anchor(score_size)

        # 对比template和search两帧图片的大小之间的区别
        s_x = s_z * (instance_size / cfg.TRACK.EXEMPLAR_SIZE)

        # 根据target中心和应该裁剪得到的search area的大小进行crop，最后resize到instance_size大小
        x_crop = self.get_subwindow(img, self.center_pos, instance_size,
                                    round(s_x), self.channel_average)

        #### 添加可视化：可视化x_crop,并将x_crop保存到images/下
        visualize = False
        if visualize:
            image = x_crop.squeeze()
            image = image.cpu().detach().numpy().astype(np.uint8).transpose(1, 2, 0)
            image = cv2.resize(image, (255, 255), interpolation=cv2.INTER_LINEAR)
            cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX,
                          cv2.CV_8U)  # 将response_map的图像格式从float32转成uint8-解决保存图片全黑的问题
            str0 = str(frame_id) + "_" + "x_crop"
            str1 = "/media/db/dbVolume/Datasets/pysot/images/images/" + str0 + ".jpg"
            cv2.imwrite(str1, image)  # 保存图片
            ####?cv2.imshow(str0, image)
            ####?cv2.waitKey(1000)

        outputs = self.model.track(x_crop, frame_id=frame_id)
        score = self._convert_score(outputs['cls'])
        pred_bbox = self._convert_bbox(outputs['loc'], anchors)

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(self.size[0] * scale_z, self.size[1] * scale_z)))
        # ratio penalty
        r_c = change((self.size[0] / self.size[1]) /
                     (pred_bbox[2, :] / pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        pscore = penalty * score

        # window
        if not self.longterm_state:
            pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
                    window * cfg.TRACK.WINDOW_INFLUENCE
        else:
            pscore = pscore * (1 - 0.001) + window * 0.001
        best_idx = np.argmax(pscore)

        bbox = pred_bbox[:, best_idx] / scale_z
        lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR

        best_score = score[best_idx]
        if best_score >= cfg.TRACK.CONFIDENCE_LOW:
            cx = bbox[0] + self.center_pos[0]
            cy = bbox[1] + self.center_pos[1]

            width = self.size[0] * (1 - lr) + bbox[2] * lr
            height = self.size[1] * (1 - lr) + bbox[3] * lr
        else:
            cx = self.center_pos[0]
            cy = self.center_pos[1]

            width = self.size[0]
            height = self.size[1]

        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        cx, cy, width, height = self._bbox_clip(cx, cy, width,
                                                height, img.shape[:2])
        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]

        if best_score < cfg.TRACK.CONFIDENCE_LOW:
            self.longterm_state = True
        elif best_score > cfg.TRACK.CONFIDENCE_HIGH:
            self.longterm_state = False

        return {
                'bbox': bbox,
                'best_score': best_score
               }
