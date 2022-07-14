# draw_pattern
import numpy as np
from PIL import Image

import argparse
import copy
import os
import os.path as osp
import time
import warnings

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
        default=None,
        help='the name of saved pattern picture')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    print('load config.')

    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    model.CLASSES = get_classes('coco')
    # re-load the pretrained model. newly added yaxian.
    
    pretrained_dict=torch.load(args.pre_train)
    model_dict=model.state_dict()
    pretrained_dict = pretrained_dict['state_dict']
    model_dict.update(pretrained_dict) # overwrite entries in the existing state dict
    model.load_state_dict(model_dict)
    print('pretrained model loaded from {}.'.format(args.pre_train))
    model.eval()

    if not args.pattern_name == None:
        pattern = model.backbone.patch
        pattern = pattern.detach()[0].cpu()
        pil_image = transforms.ToPILImage()(pattern)
        pil_image.save(args.pattern_name)
    

    dataset = build_dataset(cfg.data.test)
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

    result = []
    original = []
    out_patch = []
    bbox_result = []
    img_path = []
    cnt = 0
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            #print(data)# a dict
            resulti, out_patchi = model.forward_show(imgs=data['img'][0], \
                img_metas=data['img_metas'][0], \
                gt_keypoints=data['gt_keypoints'][0])
            result.append(resulti)
            out_patch.append(out_patchi)
            #bbox_result.append(model(return_loss=False, rescale=True, **data))
            original.append(data['img'][0])
            img_path.append(data['img_metas'][0].data[0][0]['filename'])

            cnt += 1
        if cnt >= 32:
            break
            print('len(out_patch)', len(out_patch))

    i = 0;
    for img in original:
        img = img[0].cpu()
        #mmcv.imwrite(img, 'results/img_{}.png'.format(str(i)))
        #print('img.size()', img.size())
        img_np = img.numpy()
        print('max img', np.max(img_np))
        print('min img', np.min(img_np))
        pil_image = transforms.ToPILImage()(img)
        pil_image.save('results/img_{}.png'.format(str(i)))
        i+=1
    print('save original image')

    i = 0;
    for img in result:
        img = img[0].cpu()
        #print('attack.size()', img.size())
        pil_image = transforms.ToPILImage()(img)
        pil_image.save('results/attack_{}.png'.format(str(i)))
        i+=1
    print('save attack image')

    i = 0;
    for img in out_patch:
        img = img[0][0].cpu()
        #print('patch.size()', img.size())
        #print('img', img)
        pil_image = transforms.ToPILImage()(img)
        pil_image.save('results/warped_pattern_{}.png'.format(str(i)))
        i+=1
    print('save warped_pattern')


    i = 0
    for bbox in bbox_result:
        bbox = bbox[0]
        img = img_path[i]
        model.show_result(img, bbox, score_thr=0.3,
            out_file="results/bbox_{}.png".format(str(i)))
        i += 1
    print('save bbox results')



if __name__ == "__main__":
    # execute only if run as a script
    main()
