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
from mmdet.models.backbones.patch_func import save_patch

my_img_w, my_img_h = 100, 100
patch_x, patch_y = 0., 0.
patch_w, patch_h = 120., 120.

def find_one_index(a, v=1):
    """
    return the first/second index of 1 appeared in list a
    """
    cnt = 0
    index = []
    for i in range(len(a)):
        if(cnt == 2): break;
        if a[i] == v:
            cnt += 1
            index.append(i)
    return index



def create_patch_mask(in_features, my_patch, patch_size):
    width = in_features.size(1)
    height = in_features.size(2)
    patch_mask = torch.zeros([3, width, height])
    p_w = patch_size + patch_x
    p_h = patch_size + patch_y
    patch_mask[:, int(patch_x):int(p_w), int(patch_y):int(p_h)]= 1

    return patch_mask

def create_patch_mask_bbox(im_data, bbox, advpatch):
    width = im_data.size(1)
    height = im_data.size(2)
    patch_mask = torch.zeros([3,width,height])

    p_w = bbox[2]-bbox[0]
    p_h = bbox[3]-bbox[1]
    patch_mask[:, 0:p_w,0:p_h]=1
    return patch_mask

def create_patch_mask_points(im_data, points, advpatch):
    #print('create_patch_mask_points')
    width = im_data.size(1)
    height = im_data.size(2)
    #patch_mask = torch.zeros([3,width,height])
    patch_mask = np.zeros([width, height, 3])
    
    cv2.polylines(patch_mask, np.int32([points]), 1, 1)
    cv2.fillPoly(patch_mask, np.int32([points]), (1,1,1))
    patch_mask = np.transpose(patch_mask, [2,0,1])
    patch_mask = torch.from_numpy(patch_mask)
    return patch_mask


def create_img_mask(in_features, patch_mask):
    mask = torch.ones([3,in_features.size(1), in_features.size(2)])
    img_mask = mask - patch_mask
    return img_mask

# add a patch to the original image
def add_patch(in_features, my_patch, points=None):
    """
    box[list(4)], 4 points in the image  
    """
    
    # in_features: [1,3,416,416]
    #in_features = cv2.resize(in_features, (416,416))
    patch_size = int(patch_w-patch_x)
    if points == None:
        patch_mask = create_patch_mask(in_features, my_patch, patch_size)
    else:
        patch_mask = create_patch_mask_points(in_features, points, my_patch)
    img_mask = create_img_mask(in_features, patch_mask)
    patch_mask = patch_mask.cuda()
    img_mask = img_mask.cuda()
    in_features = in_features.cuda()
    my_patch = my_patch.cuda()
    with_patch = in_features * img_mask + my_patch * patch_mask
    
    return with_patch


#four anchor points in (x,y) format, added by yaxian
def warp(src, upper_left, upper_right, bottom_left, bottom_right):
    dstTri = np.array([[upper_left[0],upper_left[1]], [upper_right[0],upper_right[1]], \
                      [bottom_left[0],bottom_left[1]], [bottom_right[0],bottom_right[1]]]).astype(np.float32)
    srcTri = np.array(
        [[0, 0], [0, src.shape[1] - 1], [src.shape[0] - 1, 0], [src.shape[0] - 1, src.shape[1] - 1]]).astype(np.float32)
    warp_mat = cv2.getPerspectiveTransform(srcTri, dstTri)
    warp_dst = cv2.warpPerspective(src, warp_mat, (416, 416))
    return warp_dst

# new function by yaxian
def add_pattern(in_img, pattern):
    # in_img is [3,416,416]
    print('in_img.shape ', in_img.shape)
    print('pattern.shape ', pattern.shape)
    width = in_img.size(1)
    height = in_img.size(2)
    pattern_mask = torch.ones([3, width, height])
    for c in range(0,3):
        for i in range(0, width):
            for j in range(0, height):
                if not (pattern[c][i][j]==0):
                    clip[i][j] = 0.0

    with_patch = in_img * img_mask + pattern
    
    return with_patch
    




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
class Darknet2(nn.Module):
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
                 patch_path=None,
                 patch_size=my_img_h,
                 patch_init_path='/home/yaxian_li/expriment/mmdetection-yax/texture/pattern_random.png',
                 training=True):
        super(Darknet2, self).__init__()
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

        self.patch_size = patch_size
        # yaxian, fix the parameter
        for p in self.parameters():
            p.requires_grad = False

        ''' define a patch here, change NO.1'''
        # training
        self.training = training
        if training:
            if patch_init_path == '':
                print('patch init from random.')
                self.patch = nn.Parameter(torch.rand(1,3, patch_size, patch_size), requires_grad=True)
            else:
                print('patch init from pic {}'.format(patch_init_path))
                lena = Image.open(patch_init_path)
                lena = lena.resize((patch_size, patch_size))
                lena = transforms.ToTensor()(lena)
                lena = lena.unsqueeze(0)
                self.patch = nn.Parameter(lena, requires_grad=True)
                
        # testing
        else:
            print('testing, load patch from path: {}'.format(patch_path))
            lena = Image.open(patch_path)
            lena = transforms.ToTensor()(lena)
            lena = lena.unsqueeze(0)
            self.patch = lena.cuda()
            

        self.norm_eval = norm_eval
        self.patch_save_cnt = -1
        self.patch_path = patch_path

        
        #self.warper = tgm.HomographyWarper(416, 416) # height, width

        # stn, yaxian
        # Spatial transformer localization-network
        '''
        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7), # in_channels=3
            nn.MaxPool2d(4, stride=4),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(4, stride=4),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 24 * 24, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        '''

    def stn(self, x):

        xs = self.localization(x)
        xs = xs.view(-1, 10 * 24 * 24)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x


    def forward(self, x, gt_keypoints=None):
        if self.training:
            self.patch_save_cnt += 1
            if self.patch_save_cnt % 100 == 0:
                save_patch_path = self.patch_path.split('.')[0] + f'_{self.patch_save_cnt}.png'
                save_patch(self.patch, save_patch_path)

        outs = []
        # self.patch = nn.Parameter(torch.clip(self.patch, min=0, max=1), requires_grad=True)
        for k in range(x.size(0)):
            warped_patch = self.warp_patch(x[k], self.patch, gt_keypoints[k])
            warped_patch = torch.clamp(warped_patch, min=0, max=1)
            box = self.get_ploy(gt_keypoints[k])
            x[k] = add_patch(x[k], warped_patch, box)
            # x[k] = torch.clip(x[k], min=0, max=1)

        for i, layer_name in enumerate(self.cr_blocks):
            cr_block = getattr(self, layer_name)
            x = cr_block(x)
            if i in self.out_indices:
                outs.append(x)

        return tuple(outs)
    
    def get_ploy(self, gt_keypoints):
        rank = [6,5,4,3,2,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
        points = []
        for i in rank:
            if(not gt_keypoints[3*(i-1) + 2]==0):
                points.append((gt_keypoints[3*(i-1)], gt_keypoints[3*(i-1) + 1]))
        return points
    
    def attacked_image(self, x, gt_keypoints=None):
        assert not (gt_keypoints == None)
        #print('attacked_image')
        x = x.cuda()
        out_patch = []
        #self.patch = self.patch.cuda()
        for k in range(x.size(0)):
            self.warped_patch = self.warp_patch(x[k], self.patch, gt_keypoints[k])
            out_patch.append(self.warped_patch)
            box = self.get_ploy(gt_keypoints[k])
            x[k] = add_patch(x[k], self.warped_patch, box)
        return x, out_patch
    
    def attacked_image_single(self, x, gt_keypoints=None):
        assert not (gt_keypoints == None)
        #print('attacked_image_single')
        outs = []
        #self.warped_patch = self.stn(self.patch)

        box = self.get_square(gt_keypoints)
        
        x = add_patch(x, self.patch, box)
        return x

    def get_square_min(self, gt_keypoints):
        minx, miny = 9999, 9999
        maxx, maxy = 0,0
        flag = True
        for i in range(int(len(gt_keypoints)/3)):
            if(gt_keypoints[3*i + 2] == 0):
                continue
            flag = False
            curx = gt_keypoints[3*i]
            cury = gt_keypoints[3*i + 1]
            if(curx < minx):
                minx = curx
            if(curx > maxx):
                maxx = curx
            if(cury < miny):
                miny = cury
            if(cury > maxy):
                maxy = cury
        if(flag):
            return (0, 0, 416, 416)
        return int(minx), int(miny), int(maxx), int(maxy)

    def get_square(self, gt_keypoints):
        upper_left, upper_right, bottom_left, bottom_right = [160,160], [160,260], [260,260], [260,160]
        if len(gt_keypoints) < 100: #coco
            left_n = [2, 3, 4, 5]
            right_n = [7, 8, 9, 10]
        else: #fashion
            left_n = [12, 13, 14, 15]
            right_n = [17, 18, 19, 20]
        flag_l, flag_r = [], []

        for i in left_n:
            if (gt_keypoints[3*(i-1) + 2] == 0):
                flag_l.append(0)  
            else:
                flag_l.append(1)
        for i in right_n:
            if (gt_keypoints[3*(i-1) + 2] == 0):
                flag_r.append(0)
            else:
                flag_r.append(1)

        #print(flag_l)
        #print(flag_r)

        if sum(flag_l) >= 2: 
            pos, pos2 = find_one_index(flag_l)
            #print(pos, pos2)
            i = left_n[pos]
            upper_left = [gt_keypoints[3*(i-1)], gt_keypoints[3*(i-1) + 1]]
            i = left_n[pos2]
            bottom_left = [gt_keypoints[3*(i-1)], gt_keypoints[3*(i-1) + 1]]

        if sum(flag_r) >= 2: 
            pos, pos2 = find_one_index(flag_r)
            i = right_n[pos]
            bottom_right = [gt_keypoints[3*(i-1)], gt_keypoints[3*(i-1) + 1]]
            i = right_n[pos2]
            upper_right = [gt_keypoints[3*(i-1)], gt_keypoints[3*(i-1) + 1]]

        return [upper_left, upper_right, bottom_right, bottom_left]
    
    def warp_patch(self, im_feature, src, gt_keypoints):
        src = src.cuda()
        '''
        factor = [
            [0.8, 0.9],
            [0.7, 0.7],
            [0.9, 0.8],
            [1.2, 1.1]
        ]
        tmp_rand = random.randint(0, len(factor)-1)
        src = F.interpolate(src,scale_factor=factor[tmp_rand])
        '''
        minx, miny, maxx, maxy = self.get_square_min(gt_keypoints)
        dstTri = np.array([[
            [minx, miny], [minx, maxy],
            [maxx, miny], [maxx, maxy]
        ]])
        srcTri = np.array(
            [[[0, 0], [0, int(src.size(3)) - 1], [int(src.size(2)) - 1, 0],
            [int(src.size(2)) - 1, int(src.size(3)) - 1]]]).astype(np.float32)
        srcTri = torch.from_numpy(srcTri).float().cuda()
        dstTri = torch.from_numpy(dstTri).float().cuda()
       
        try:
            warp_mat = tgm.get_perspective_transform(srcTri, dstTri)
        except:
            dstTri = np.array([[[0, 0], 
                [0, int(im_feature.size(2)) - 1], [int(im_feature.size(1)) - 1, 0],
                [int(im_feature.size(1)) - 1, int(im_feature.size(2)) - 1]]]).astype(np.float32)
            dstTri = torch.from_numpy(dstTri).float().cuda()
            warp_mat = tgm.get_perspective_transform(srcTri, dstTri)
        
        output = tgm.warp_perspective(src, warp_mat, dsize=(im_feature.size(1), im_feature.size(2)))
        return output


        

        
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
        super(Darknet2, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

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
