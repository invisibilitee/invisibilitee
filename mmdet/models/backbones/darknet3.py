# Copyright (c) 2019 Western Digital Corporation or its affiliates.
# attack the yolo in backbone darknet
import logging

import torch
import cv2
import torchgeometry as tgm
import numpy as np
import torch.nn.functional as F
from PIL import Image
import random
import os

import torch.nn as nn
import torchvision.transforms as transforms
from mmcv.cnn import ConvModule, constant_init, kaiming_init
from mmcv.runner import load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm

from ..builder import BACKBONES
from mmdet.models.backbones.patch_func import *


class ResBlock(nn.Module):
    """The basic residual block used in Darknet. Each ResBlock consists of two
    ConvModules and the input is added to the final output. Each ConvModule is
    composed of Conv, BN, and LeakyReLU. In YoloV3 paper, the first convLayer
    has half of the number of the filters as much as the second convLayer. The
    first convLayer has filter size of 1x1 and the second one has the filter
    size of 3x3.

    Args:
        in_channels (int): The input channels. Must be even.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: dict(type='BN', requires_grad=True)
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='LeakyReLU', negative_slope=0.1).
    """

    def __init__(self,
                 in_channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.1)):
        super(ResBlock, self).__init__()
        assert in_channels % 2 == 0  # ensure the in_channels is even
        half_in_channels = in_channels // 2

        # shortcut
        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.conv1 = ConvModule(in_channels, half_in_channels, 1, **cfg)
        self.conv2 = ConvModule(
            half_in_channels, in_channels, 3, padding=1, **cfg)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + residual

        return out


@BACKBONES.register_module()
class Darknet3(nn.Module):
    """Darknet backbone with patch. liyaxian.

    Args:
        depth (int): Depth of Darknet. Currently only support 53.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Default: -1.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: dict(type='BN', requires_grad=True)
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='LeakyReLU', negative_slope=0.1).
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.

    Example:
        >>> from mmdet.models import Darknet
        >>> import torch
        >>> self = Darknet(depth=53)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 416, 416)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        ...
        (1, 256, 52, 52)
        (1, 512, 26, 26)
        (1, 1024, 13, 13)
    """

    # Dict(depth: (layers, channels))
    arch_settings = {
        53: ((1, 2, 8, 8, 4), ((32, 64), (64, 128), (128, 256), (256, 512),
                               (512, 1024)))
    }

    def __init__(self,
                 depth=53,
                 out_indices=(3, 4, 5),
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
                 norm_eval=True,

                 patch_ori_size=100,
                 patch_size=100,
                 patch_init_path='',
                 patch_path=None, 

                 training=True):
        super(Darknet3, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for darknet')
        self.depth = depth
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.layers, self.channels = self.arch_settings[depth]

        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.conv1 = ConvModule(3, 32, 3, padding=1, **cfg)

        self.cr_blocks = ['conv1']
        for i, n_layers in enumerate(self.layers):
            layer_name = f'conv_res_block{i + 1}'
            in_c, out_c = self.channels[i]
            self.add_module(
                layer_name,
                self.make_conv_res_block(in_c, out_c, n_layers, **cfg))
            self.cr_blocks.append(layer_name)

        # yaxian, fix the parameter
        for p in self.parameters():
            p.requires_grad = False

        ''' define a patch here, change NO.1'''
        ##########################
        self.patch_ori_size = patch_ori_size
        self.patch_size     = patch_size
        self.training       = training
        self.patch_path     = patch_path
        
        # yaxian, fix the parameter
        for p in self.parameters():
            p.requires_grad = False

        # add patch
        self.patch = get_patch_tensor(self.patch_path,
                                      training=self.training, 
                                      patch_init_path=patch_init_path,
                                      patch_size=(self.patch_ori_size, self.patch_ori_size))
        ###########################

        self.norm_eval = norm_eval
        self.patch_save_cnt = 0

    def forward(self, x, gt_keypoints=None):
        # Add Patch 2021年05月11日
        transformed_patch, x = before_forward_downsample(x, self.patch_size, self.patch, gt_keypoints)

        if self.training:
            self.patch_save_cnt += 1
            if self.patch_save_cnt % 100 == 0:
                save_patch_path = self.patch_path.split(
                    '.')[0] + f'_{self.patch_save_cnt}.png'
                save_patch(self.patch, save_patch_path)

                save_patch_path = self.patch_path.split(
                    '.')[0] + f'_{self.patch_save_cnt}_down.png'
                save_patch(transformed_patch, save_patch_path)


        outs = []
        for i, layer_name in enumerate(self.cr_blocks):
            cr_block = getattr(self, layer_name)
            x = cr_block(x)
            if i in self.out_indices:
                outs.append(x)

        return tuple(outs)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            # yaxian
            #load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            print('weight init!')
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)

        else:
            raise TypeError('pretrained must be a str or None')

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for i in range(self.frozen_stages):
                m = getattr(self, self.cr_blocks[i])
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def train(self, mode=True):
        super(Darknet3, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
    
    def attacked_image(self, x, gt_keypoints=None):
        assert not (gt_keypoints == None)
        #print('attacked_image')
        transformed_patch = F.adaptive_avg_pool2d(self.patch,
            (self.patch_size, self.patch_size))

        x = x.cuda()
        out_patch = []
        #self.patch = self.patch.cuda()
        for k in range(x.size(0)):
            warped_patch = warp_patch(x[k], transformed_patch, gt_keypoints[k])
            out_patch.append(warped_patch)
            box = get_ploy(gt_keypoints[k])
            x[k] = add_patch(x[k], warped_patch, box)
        return x, out_patch

    @staticmethod
    def make_conv_res_block(in_channels,
                            out_channels,
                            res_repeat,
                            conv_cfg=None,
                            norm_cfg=dict(type='BN', requires_grad=True),
                            act_cfg=dict(type='LeakyReLU',
                                         negative_slope=0.1)):
        """In Darknet backbone, ConvLayer is usually followed by ResBlock. This
        function will make that. The Conv layers always have 3x3 filters with
        stride=2. The number of the filters in Conv layer is the same as the
        out channels of the ResBlock.
190
        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            res_repeat (int): The number of ResBlocks.
            conv_cfg (dict): Config dict for convolution layer. Default: None.
            norm_cfg (dict): Dictionary to construct and config norm layer.
                Default: dict(type='BN', requires_grad=True)
            act_cfg (dict): Config dict for activation layer.
                Default: dict(type='LeakyReLU', negative_slope=0.1).
        """

        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

        model = nn.Sequential()
        model.add_module(
            'conv',
            ConvModule(
                in_channels, out_channels, 3, stride=2, padding=1, **cfg))
        for idx in range(res_repeat):
            model.add_module('res{}'.format(idx),
                             ResBlock(out_channels, **cfg))
        return model
