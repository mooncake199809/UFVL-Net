# Copyright (c) OpenMMLab. All rights reserved.
import os
import sys
import cv2
import numpy as np

import torch
import torch.nn as nn
from mmcv.runner import BaseModule

from ..builder import ARCHITECTURE, build_backbone, build_head
from ..utils.augment import Augments
from .base import BaseClassifier

@ARCHITECTURE.register_module()
class FDANET(BaseClassifier):
    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 pretrained=None,
                 train_cfg=None,
                 init_cfg=None,
                 dataset="7Scenes"):
        super(FDANET, self).__init__(init_cfg)

        self.backbone = build_backbone(backbone)
        self.head = build_head(head)
        self.loss = EuclideanLoss_with_Uncertainty()
        self.dataset = dataset

    def forward_test(self, img, gt_lables=None, **kwargs):
        import numpy as np
        import sys
        import os
        import cv2
        sys.path.append("/home/dk/ufvl_net/ufvl_net/models/architecture/pnpransac")
        import pnpransac
        if self.dataset == "7Scenes":
            intrinsics_color = np.array([[525.0, 0.0,     320.0],
                           [0.0,     525.0, 240.0],
                           [0.0,     0.0,  1.0]])
        elif self.dataset == "12Scenes":
            intrinsics_color = np.array([[572.0, 0.0,     320.0],
                           [0.0,     572.0, 240.0],
                           [0.0,     0.0,  1.0]])
        pose_solver = pnpransac.pnpransac(intrinsics_color[0, 0], intrinsics_color[1, 1], intrinsics_color[0, 2],
                                          intrinsics_color[1, 2])

        def get_pose_err(pose_gt, pose_est):
            transl_err = np.linalg.norm(pose_gt[0:3, 3] - pose_est[0:3, 3])
            rot_err = pose_est[0:3, 0:3].T.dot(pose_gt[0:3, 0:3])
            rot_err = cv2.Rodrigues(rot_err)[0]  # 旋转向量 [3 1]
            rot_err = np.reshape(rot_err, (1, 3))  # 旋转向量 [1 3]
            rot_err = np.reshape(np.linalg.norm(rot_err, axis=1), -1) / np.pi * 180.  # 二范数即转角
            return transl_err, rot_err[0]

        x = np.linspace(4, 640 - 4, 80)
        y = np.linspace(4, 480 - 4, 60)

        xx, yy = np.meshgrid(x, y)  # [60 80]
        pcoord = np.concatenate((np.expand_dims(xx, axis=2), np.expand_dims(yy, axis=2)), axis=2)

        x = self.backbone(img)
        coord, uncertainty = self.head(x[0])
        coord = np.transpose(coord.cpu().data.numpy()[0, :, :, :], (1, 2, 0))  # [3 60 80]->[60 80 3]
        uncertainty = np.transpose(uncertainty[0].cpu().data.numpy(), (1, 2, 0))
        coord = np.concatenate([coord, uncertainty], axis=2)  # [60 80 4]
        coord = np.ascontiguousarray(coord)
        pcoord = np.ascontiguousarray(pcoord)

        pcoord = pcoord.reshape(-1, 2)
        coords = coord[:, :, 0:3].reshape(-1, 3)
        confidences = coord[:, :, 3].flatten().tolist()

        coords_filtered = []
        coords_filtered_2D = []
        for i in range(len(confidences)):
            if confidences[i] > 0:
                coords_filtered.append(coords[i])
                coords_filtered_2D.append(pcoord[i])

        coords_filtered = np.vstack(coords_filtered)
        coords_filtered_2D = np.vstack(coords_filtered_2D)

        rot, transl = pose_solver.RANSAC_loop(coords_filtered_2D.astype(np.float64), coords_filtered.astype(np.float64),
                                              256)  # 预测结果,每次取256组点进行PNP Tcw
        pose_gt = gt_lables.cpu().numpy()[0, :, :]  # [4 4]
        pose_est = np.eye(4)  # [4 4]
        pose_est[0:3, 0:3] = cv2.Rodrigues(rot)[0].T  # Rwc
        pose_est[0:3, 3] = -np.dot(pose_est[0:3, 0:3], transl)  # twc

        transl_err, rot_err = get_pose_err(pose_gt, pose_est)

        return dict(trans_error_med=transl_err, rot_err_med=rot_err)

    def forward_train(self, img, gt_lables, mask, **kwargs):
        x = self.backbone(img)
        coord, uncer = self.head(x[0])
        loss, accuracy = self.loss(coord, gt_lables, mask, uncer)
        losses = dict(loss=loss, accuracy=accuracy)
        return losses

    def forward(self, img, gt_lables, mask=None, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.

        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_meta are single-nested (i.e. Tensor and
        List[dict]), and when `resturn_loss=False`, img and img_meta should be
        double nested (i.e.  List[Tensor], List[List[dict]]), with the outer
        list indicating test time augmentations.
        """
        if return_loss:
            assert mask is not None
            return self.forward_train(img, gt_lables, mask, **kwargs)
        else:
            return self.forward_test(img, gt_lables, **kwargs)
    
    def extract_feat(self, img, stage='neck'):
        pass

    def simple_test(self, img, img_metas=None, **kwargs):
        pass


class EuclideanLoss_with_Uncertainty(nn.Module):
    def __init__(self):
        super(EuclideanLoss_with_Uncertainty, self).__init__()
        self.pdist = nn.PairwiseDistance(p=2)   

    def forward(self, pred, target, mask, certainty):
        loss_reg = self.pdist(pred.permute(0,2,3,1), target.permute(0,2,3,1))  
        certainty_map = torch.clamp(certainty, 1e-6)  
        loss_map = 3 * torch.log(certainty_map) + loss_reg / (2 * certainty_map.pow(2))

        loss_map = loss_map * mask 
        loss =torch.sum(loss_map) / mask.sum()

        if mask is not None:
            valid_pixel = mask.sum() + 1
            diff_coord_map = mask * loss_reg

        thres_coord_map = torch.clamp(diff_coord_map - 0.05, 0)
        num_accurate = valid_pixel - thres_coord_map.nonzero().shape[0]
        accuracy = num_accurate / valid_pixel
        loss1 = torch.sum(loss_reg*mask) / mask.sum()
        return loss, accuracy
