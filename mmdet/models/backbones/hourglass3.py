import torch.nn as nn
from mmcv.cnn import ConvModule

import cv2
import torch
import torchgeometry as tgm
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from mmdet.models.backbones.patch_func import *

from ..builder import BACKBONES
from ..utils import ResLayer
from .resnet import BasicBlock

'''
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
    if points == None:
        patch_size = int(patch_w-patch_x)
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
'''

class HourglassModule(nn.Module):
    """Hourglass Module for HourglassNet backbone.

    Generate module recursively and use BasicBlock as the base unit.

    Args:
        depth (int): Depth of current HourglassModule.
        stage_channels (list[int]): Feature channels of sub-modules in current
            and follow-up HourglassModule.
        stage_blocks (list[int]): Number of sub-modules stacked in current and
            follow-up HourglassModule.
        norm_cfg (dict): Dictionary to construct and config norm layer.
    """

    def __init__(self,
                 depth,
                 stage_channels,
                 stage_blocks,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super(HourglassModule, self).__init__()

        self.depth = depth

        cur_block = stage_blocks[0]
        next_block = stage_blocks[1]

        cur_channel = stage_channels[0]
        next_channel = stage_channels[1]

        self.up1 = ResLayer(
            BasicBlock, cur_channel, cur_channel, cur_block, norm_cfg=norm_cfg)

        self.low1 = ResLayer(
            BasicBlock,
            cur_channel,
            next_channel,
            cur_block,
            stride=2,
            norm_cfg=norm_cfg)

        if self.depth > 1:
            self.low2 = HourglassModule(depth - 1, stage_channels[1:],
                                        stage_blocks[1:])
        else:
            self.low2 = ResLayer(
                BasicBlock,
                next_channel,
                next_channel,
                next_block,
                norm_cfg=norm_cfg)

        self.low3 = ResLayer(
            BasicBlock,
            next_channel,
            cur_channel,
            cur_block,
            norm_cfg=norm_cfg,
            downsample_first=False)

        self.up2 = nn.Upsample(scale_factor=2)

    def forward(self, x):
        """Forward function."""
        up1 = self.up1(x)
        low1 = self.low1(x)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2 = self.up2(low3)
        return up1 + up2


@BACKBONES.register_module()
class HourglassNet3(nn.Module):
    # yaxian changed
    """HourglassNet backbone.

    Stacked Hourglass Networks for Human Pose Estimation.
    More details can be found in the `paper
    <https://arxiv.org/abs/1603.06937>`_ .

    Args:
        downsample_times (int): Downsample times in a HourglassModule.
        num_stacks (int): Number of HourglassModule modules stacked,
            1 for Hourglass-52, 2 for Hourglass-104.
        stage_channels (list[int]): Feature channel of each sub-module in a
            HourglassModule.
        stage_blocks (list[int]): Number of sub-modules stacked in a
            HourglassModule.
        feat_channel (int): Feature channel of conv after a HourglassModule.
        norm_cfg (dict): Dictionary to construct and config norm layer.

    Example:
        >>> from mmdet.models import HourglassNet
        >>> import torch
        >>> self = HourglassNet()
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 511, 511)
        >>> level_outputs = self.forward(inputs)
        >>> for level_output in level_outputs:
        ...     print(tuple(level_output.shape))
        (1, 256, 128, 128)
        (1, 256, 128, 128)
    """

    def __init__(self,
                 downsample_times=5,
                 num_stacks=2,
                 stage_channels=(256, 256, 384, 384, 384, 512),
                 stage_blocks=(2, 2, 2, 2, 2, 4),
                 feat_channel=256,
                 norm_cfg=dict(type='BN', requires_grad=True),
                
                 patch_ori_size=100,
                 patch_size=100,
                 patch_init_path='',
                 patch_path=None, 
                 
                 training=True):
        super(HourglassNet3, self).__init__()

        self.num_stacks = num_stacks
        assert self.num_stacks >= 1
        assert len(stage_channels) == len(stage_blocks)
        assert len(stage_channels) > downsample_times

        cur_channel = stage_channels[0]

        self.stem = nn.Sequential(
            ConvModule(3, 128, 7, padding=3, stride=2, norm_cfg=norm_cfg),
            ResLayer(BasicBlock, 128, 256, 1, stride=2, norm_cfg=norm_cfg))

        self.hourglass_modules = nn.ModuleList([
            HourglassModule(downsample_times, stage_channels, stage_blocks)
            for _ in range(num_stacks)
        ])

        self.inters = ResLayer(
            BasicBlock,
            cur_channel,
            cur_channel,
            num_stacks - 1,
            norm_cfg=norm_cfg)

        self.conv1x1s = nn.ModuleList([
            ConvModule(
                cur_channel, cur_channel, 1, norm_cfg=norm_cfg, act_cfg=None)
            for _ in range(num_stacks - 1)
        ])

        self.out_convs = nn.ModuleList([
            ConvModule(
                cur_channel, feat_channel, 3, padding=1, norm_cfg=norm_cfg)
            for _ in range(num_stacks)
        ])

        self.remap_convs = nn.ModuleList([
            ConvModule(
                feat_channel, cur_channel, 1, norm_cfg=norm_cfg, act_cfg=None)
            for _ in range(num_stacks - 1)
        ])

        self.relu = nn.ReLU(inplace=True)

        # yaxian
        self.training = training
        # self.warper = tgm.HomographyWarper(416, 416) # height, width
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
        self.patch_save_cnt = 0

        

    def init_weights(self, pretrained=None):
        """Init module weights.

        We do nothing in this function because all modules we used
        (ConvModule, BasicBlock and etc.) have default initialization, and
        currently we don't provide pretrained model of HourglassNet.

        Detector's __init__() will call backbone's init_weights() with
        pretrained as input, so we keep this function.
        """
        # Training Centripetal Model needs to reset parameters for Conv2d
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.reset_parameters()

    def forward(self, x, gt_keypoints=None):
        # Add Patch 2021年05月11日
        transformed_patch, x = before_forward_downsample(x, self.patch_size, self.patch, gt_keypoints)

        # if self.training:
        #     self.patch_save_cnt += 1
        #     if self.patch_save_cnt % 100 == 0:
        #         save_patch_path = self.patch_path.split(
        #             '.')[0] + f'_{self.patch_save_cnt}.png'
        #         save_patch(self.patch, save_patch_path)

        #         save_patch_path = self.patch_path.split(
        #             '.')[0] + f'_{self.patch_save_cnt}_down.png'
        #         save_patch(transformed_patch, save_patch_path)

        inter_feat = self.stem(x)
        out_feats = []

        for ind in range(self.num_stacks):
            single_hourglass = self.hourglass_modules[ind]
            out_conv = self.out_convs[ind]

            hourglass_feat = single_hourglass(inter_feat)
            out_feat = out_conv(hourglass_feat)
            out_feats.append(out_feat)

            if ind < self.num_stacks - 1:
                inter_feat = self.conv1x1s[ind](
                    inter_feat) + self.remap_convs[ind](
                        out_feat)
                inter_feat = self.inters[ind](self.relu(inter_feat))

        return out_feats


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

    def warp_patch(self, im_feature, src, gt_keypoints):
        #print('im_feature.size()',im_feature.size())
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
            #return src
        
        output = tgm.warp_perspective(src, warp_mat, dsize=(im_feature.size(1), im_feature.size(2)))
        return output

    def get_square(self, gt_keypoints):
        upper_left, upper_right, bottom_left, bottom_right = [160,160], [160,260], [260,160], [260,260]
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