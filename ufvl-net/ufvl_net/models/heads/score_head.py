import math
import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule

from ..builder import HEADS


@HEADS.register_module()
class RegHead(BaseModule):
    def __init__(self,
                in_channel=2048,
                norm_cfg=dict(type='BN'),
                init_cfg=None):
        super(RegHead, self).__init__(init_cfg=init_cfg)

        self.in_channel = in_channel
        self.conv_reg1 = ConvModule(self.in_channel, 
                                    512, 
                                    kernel_size=3,
                                    padding=1,
                                    norm_cfg=norm_cfg)
        self.conv_reg2 = ConvModule(512, 
                                    256, 
                                    kernel_size=3, 
                                    padding=1, 
                                    norm_cfg=norm_cfg)
        self.conv_reg3 = ConvModule(256, 
                                    128, 
                                    kernel_size=3, 
                                    padding=1,
                                    norm_cfg=norm_cfg)
        self.coord_conv = ConvModule(128, 
                                     64, 
                                     kernel_size=3, 
                                     norm_cfg=norm_cfg,
                                     padding=1)
        
        self.coord_reg = torch.nn.Conv2d(64, 3, kernel_size=1)

        self.uncer_conv = ConvModule(128, 
                                     64, 
                                     kernel_size=3, 
                                     norm_cfg=norm_cfg,
                                     padding=1)
        
        self.uncer_reg = torch.nn.Conv2d(64, 1, kernel_size=1)


    def forward(self, feat, **kwargs):

        feat = self.conv_reg3(self.conv_reg2(self.conv_reg1(feat)))
        coord = self.coord_reg(self.coord_conv(feat))
        uncer = self.uncer_reg(self.uncer_conv(feat))
        uncer = torch.sigmoid(uncer)
        return coord, uncer