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
        default='result',
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



def main():
    args = parse_args()

    if not os.path.exists(args.result_root):
        os.mkdir(args.result_root)

    cfg = Config.fromfile(args.config)
    print('load config.')

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
        workers_per_gpu=1,
        dist=distributed,
        shuffle=False)

    original    = []

    for i, data in enumerate(tqdm(data_loader)):
        original.append(data['img_metas'][0].data[0][0]['filename'])
        if i >= args.limit:
            break

 
    attacked_path = []
    for i, imgname in enumerate(tqdm(original)):
        imgpa = f'{args.result_root}/attack_{i}.png'
        attacked_path.append(imgpa)

    # infer bbox
    print('save attack bbox results')
    infer_model = init_detector('configs/yolo/yolov3_noattack_fashion.py',\
         'checkpoints/yolov3_d53_mstrain-608_273e_coco-139f5633.pth', device='cuda')
    for i, at_img in enumerate(tqdm(attacked_path)):
        result = inference_detector(infer_model, at_img)
        infer_model.show_result(at_img, result, score_thr=0.5,\
             out_file=f'{args.result_root}/bbox_attack_{i}.png')

    # print('save original bbox results')
    # for i, img in enumerate(tqdm(original)):
    #     result = inference_detector(infer_model, img)
    #     infer_model.show_result(img, result, score_thr=0.5,\
    #          out_file=f'{args.result_root}/bbox_original_{i}.png')

if __name__ == "__main__":
    # execute only if run as a script
    main()
