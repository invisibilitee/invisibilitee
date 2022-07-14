_base_ = './retinanet_r50_fpn_1x_fashion.py'
patch_saved_path = '/home/yaxian_li/expriment/mmdetection-yax/patch_downsample_300p100/p_150400.png'

model = dict(pretrained='torchvision://resnet101', 
    backbone=dict(
        depth=101,
        patch_path=patch_saved_path, #yaxian
        training=False
        ))
