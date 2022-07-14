# draw_pattern
import numpy as np
from PIL import Image

import argparse
import copy
import os
import os.path as osp
import time
import warnings
import torch.nn as nn
import mmcv
import torch
import torchvision.transforms as transforms
from mmdet.apis import init_detector, inference_detector
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import get_git_hash
from mmdet.core import get_classes

from mmdet import __version__
from mmdet.apis import set_random_seed, train_detector
from mmdet.datasets import build_dataset, build_dataloader
from mmdet.datasets.deep_fashion import DeepFashionDataset
from mmdet.models import build_detector
from mmdet.utils import collect_env, get_root_logger

from tqdm import tqdm
from mmdet.models.backbones.patch_func import *

def parse_args():
    parser = argparse.ArgumentParser(description='Draw a pattern from detector model')
    parser.add_argument(
        '--pre_train', 
        default='work_dirs/yolov3_d53_dark2warp_mstrain-608_273e_fashion_randv2/latest.pth',
        help='pre trained model file path') # newly added by yaxian
    
    parser.add_argument(
        '--config', 
        help='train config file path',
        default='configs/yolo/yolov3_d53_dark2warp_mstrain-608_273e_fashion.py')
    
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')

    parser.add_argument(
        '--pattern_name',
        default='',
        help='the name of saved pattern picture')
    
    parser.add_argument(
        '--result_root',
        default='results',
        help='the name of result folder')

    parser.add_argument(
        '--limit',
        type=int,
        default=100,
        help='the image number')

    args = parser.parse_args()
    return args

def save_tensor_img(path, tensor):
    pil_image = transforms.ToPILImage()(tensor)
    pil_image.save(path)


def person_detection(model, img):
    flag = False
    result = inference_detector(model, img)
    if isinstance(result, tuple):
        bbox_result, segm_result = result
        if isinstance(segm_result, tuple):
            segm_result = segm_result[0]  # ms rcnn
    else:
        bbox_result, segm_result = result, None
    if not len(bbox_result[0]) == 0: # person in picture
        flag = True
    return flag

def myget_keypoints(points):
    keypoints = []
    for i in range(len(points)):
        keypoints.append(points[i][0])
        keypoints.append(points[i][1])
        keypoints.append(1)
    #print('keypoints', keypoints)
    return keypoints

def myattacked_image(patch, points):
    #x = torch.zeros(1, 745, 1000, 3)
    x = torch.zeros(1, 3000, 3000, 3)
    keypoints = myget_keypoints(points)
    print('keypoints', keypoints)
    warped_patch = warp_patch(x, patch, keypoints)
    patch_mask = create_patch_mask_points(x, points, warped_patch)
    #warped_patch = warped_patch*patch_mask.cuda()
    return warped_patch


def main():
    args = parse_args()

    if not os.path.exists(args.result_root):
        os.mkdir(args.result_root)

    cfg = Config.fromfile(args.config)
    print('load config.')

    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    model.CLASSES = get_classes('coco')
    # re-load the pretrained model. newly added yaxian.
    
    pretrained_dict=torch.load(args.pre_train, map_location={'cuda:2':'cuda'})
    model_dict=model.state_dict()
    pretrained_dict = pretrained_dict['state_dict']
    model_dict.update(pretrained_dict) # overwrite entries in the existing state dict
    model.load_state_dict(model_dict)
    print('pretrained model loaded from {}.'.format(args.pre_train))
    model.eval()

    if args.pattern_name != '':
        pattern = Image.open(args.pattern_name)
        # pattern = pattern.resize((100, 100))
        pattern = transforms.ToTensor()(pattern)
        pattern = pattern.unsqueeze(0)
        pattern = nn.Parameter(pattern, requires_grad=True)
        model.backbone.patch = pattern
    
    gt_keypoints = [
        [496, 86.5],  [420, 68.5],  [396, 17.5],
        [316, 15.5],  [241, 17.5],  [136, 75.5],
        [8, 145.5 ],  [109, 334.5], [242, 262.5],
        [243, 414.5], [242, 546.5], [242, 676.5],
        [384, 676.5], [594, 674.5], [759, 676.5],
        [759, 537.5], [758, 382.5], [760, 262.5],
        [890, 332.5], [992, 146.5], [895, 93.5],
        [757, 16.5],  [666,  16.5], [600, 18.5],
        [575, 67.5], [0,  0], [2999,  2999],
    ]
    warp_img = myattacked_image(model.backbone.patch, gt_keypoints)
    print('warp_img.size()', warp_img.size())
    print('save warped_pattern')
    save_tensor_img(f'{args.result_root}/faster_p200_257100_down.png', warp_img[0].cpu())


    '''
    dataset = build_dataset(cfg.data.test)
    print(f'Dataset: {len(dataset)}')
    print('cfg.data.test', cfg.data.test)
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, backend='nccl')
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    result      = []
    original    = []
    out_patch   = []
    img_path    = []

    for i, data in enumerate(tqdm(data_loader)):
        with torch.no_grad():
            #print(data)# a dict
            #print('data[img_metas][0]', data['img_metas'][0])
            resulti, out_patchi = model.forward_show(imgs=data['img'][0], \
                img_metas=data['img_metas'][0], \
                gt_keypoints=data['gt_keypoints'][0])
            result.append(resulti)
            out_patch.append(out_patchi)

            original.append(data['img'][0])
            img_path.append(data['img_metas'][0].data[0][0]['filename'])

        if i >= args.limit:
            break

    # print('save original image')
    # for i, img in enumerate(tqdm(original)):
    #    save_tensor_img(f'{args.result_root}/original_{i}.png', img[0].cpu())

    attacked_path = []
    print('save attack image')
    for i, img in enumerate(tqdm(result)):
        img_path = f'{args.result_root}/attack_{i}.png'
        save_tensor_img(img_path, img[0].cpu())
        attacked_path.append(img_path)
    
    print('save warped_pattern')
    for i, img in enumerate(tqdm(out_patch)):
        save_tensor_img(f'{args.result_root}/wrap_{i}.png', img[0][0].cpu())
    '''

if __name__ == "__main__":
    # execute only if run as a script
    main()
