# Mingyu Zhang
import torch
import numpy as np
import cv2
import torchgeometry as tgm
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.nn as nn

from PIL import Image
import random
import os
# mmcv/mmcv/runner/hooks/lr_updater.py
import mmcv.runner.hooks.lr_updater
# define patch size
my_img_w, my_img_h = 416, 416
patch_x, patch_y   = 0., 0.
patch_w, patch_h   = 120., 120.


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
    patch_mask = np.zeros([width, height, 3])
    
    cv2.polylines(patch_mask, np.int32([points]), 1, 1)
    # print(np.int32([points]).shape)
    cv2.fillPoly(patch_mask, np.int32([points]), (1,1,1))
    patch_mask = np.transpose(patch_mask, [2,0,1])
    patch_mask = torch.from_numpy(patch_mask)
    return patch_mask


def create_img_mask(in_features, patch_mask):
    mask = torch.ones([3,in_features.size(1), in_features.size(2)])
    img_mask = mask - patch_mask
    return img_mask

# add a patch to the original image
# ! add clamp option
def add_patch(in_features, my_patch, points=None):
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
    with_patch = torch.clamp(with_patch, min=0, max=1)
    return with_patch

def save_patch(my_patch, save_path):
    # check if path is exist
    folder_path, file_name = os.path.split(save_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    patch_img = transforms.ToPILImage()(my_patch.cpu()[0])
    patch_img.save(save_path)

def load_patch(path, patch_size):
    lena = Image.open(path)
    lena = lena.resize((patch_size[0], patch_size[1]))
    lena = transforms.ToTensor()(lena)
    lena = lena.unsqueeze(0)
    return lena

# create by mingyu
def get_patch_tensor(patch_path, 
     training=False,
     patch_init_path='',
     patch_size=(416, 416)):
    '''
    Get Trainable Patch Tensor
        Args:
            patch_path     : test: patch_path
            training       : self.training
            patch_init_path: train: patch_init_path
    '''
    # training
    if training:
        if patch_init_path == '':
            print('patch init from random.')
            patch = nn.Parameter(torch.rand(1,3, patch_size[0], patch_size[1]), requires_grad=True)
        else:
            print('patch init from pic {}'.format(patch_init_path))
            patch = load_patch(patch_init_path, patch_size)
            patch = nn.Parameter(patch, requires_grad=True)

    # testing
    else:
        print('testing, load patch from path: {}'.format(patch_path))
        patch = load_patch(patch_path, patch_size)
        patch = patch.cuda()

    return patch

# create by mingyu
def before_forward(x, patch, gt_keypoints=None):
    '''
    Add patch before forward
        Args:
            x:            x
            patch:        self.patch
            gt_keypoints: gt_keypoints
        Usage:
            def forward(x, gt_keypoints):
                x = before_forward(x, self.patch, gt_keypoints)
                
                ...other forward codes
    '''

    for k in range(x.size(0)):
        warped_patch = warp_patch(x[k], patch, gt_keypoints[k])
        box          = get_ploy(gt_keypoints[k])
        x[k]         = add_patch(x[k], warped_patch, box)
    patched_x = x
    return patched_x

# new_im = transforms.ColorJitter(brightness=1)(im)
# new_im = transforms.ColorJitter(contrast=1)(im)
# new_im = transforms.ColorJitter(saturation=0.5)(im)
# new_im = transforms.ColorJitter(hue=0.5)(im)
# new_im.save(os.path.join(outfile, '5_1.jpg'))

def before_forward_downsample_transform(x, patch_size, patch, gt_keypoints=None):
    '''
    Add patch before forward + second transform
        Args:
            x:            x
            patch_size:   target patch size
            patch:        self.patch
            gt_keypoints: gt_keypoints
        Usage:
            def forward(x, gt_keypoints):
                x = before_forward(x, self.patch, gt_keypoints)
                
                ...other forward codes
    '''
    transformed_patch = F.adaptive_avg_pool2d(patch,
            (patch_size, patch_size))

    patched_x = before_forward(x, patch, gt_keypoints)
    # second transform, yaxian 0513
    mode = random.randint(1,4)%2
    if mode == 1:
        brightness = float(1.0*random.randint(0,69)/100)
        contrast = float(1.0*random.randint(0,69)/100)
        saturation = float(1.0*random.randint(0,69)/100)
        patched_x = transforms.ColorJitter(brightness=brightness, \
            contrast=contrast, saturation=saturation, \
            hue=0.1)(patched_x)

    return transformed_patch, patched_x



# downsample 2021年05月11日
# create by mingyu
def before_forward_downsample(x, patch_size, patch, gt_keypoints=None):
    '''
    Add patch before forward
        Args:
            x:            x
            patch_size:   target patch size
            patch:        self.patch
            gt_keypoints: gt_keypoints
        Usage:
            def forward(x, gt_keypoints):
                x = before_forward(x, self.patch, gt_keypoints)
                
                ...other forward codes
    '''
    transformed_patch = F.adaptive_avg_pool2d(patch,
            (patch_size, patch_size))

    patched_x = before_forward(x, patch, gt_keypoints)
    return transformed_patch, patched_x

def get_ploy(gt_keypoints):
    if len(gt_keypoints) == 1:
        gt_keypoints = gt_keypoints[0]
    if len(gt_keypoints) > 500:
        rank = [6,5,4,3,2,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25] #deepfashion
    else:
        #rank = [1,5,2,4,13,12,7,9,8,6,10,11,3] # tf model extracted keypoints
        rank = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
    points = []
    for i in rank:
        if(not gt_keypoints[3*(i-1) + 2]==0):
            points.append((gt_keypoints[3*(i-1)], gt_keypoints[3*(i-1) + 1]))
    return points
    

def get_square_min(gt_keypoints):
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

def warp_patch(im_feature, src, gt_keypoints):
    src = src.cuda()
    minx, miny, maxx, maxy = get_square_min(gt_keypoints)
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