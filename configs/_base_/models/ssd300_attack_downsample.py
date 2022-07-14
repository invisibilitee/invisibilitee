# model settings

# 训练时 patch原始尺寸：参数数目=patch_ori_size * patch_ori_size * 3
patch_ori_size = 224

# 训练/测试 时patch目标尺寸：将patch downsample到目标尺寸后加入原图
patch_size=100

# 训练时 中间patch存储目录
patch_patch = '/home/yaxian_li/expriment/mmdetection-yax/patch_downsample_224p100/p_2100.png'

input_size = 300
model = dict(
    type='SingleStageDetector',
    #pretrained='open-mmlab://vgg16_caffe',
    backbone=dict(
        type='SSDVGG3',
        input_size=input_size,
        depth=16,
        with_last_pool=False,
        ceil_mode=True,
        out_indices=(3, 4),
        out_feature_indices=(22, 34),
        l2_norm_scale=20,
        patch_path      = patch_patch,
        patch_init_path = patch_patch,
        patch_size      = patch_size,
        patch_ori_size  = patch_ori_size,
        training=False),
    neck=None,
    bbox_head=dict(
        type='SSDHead',
        in_channels=(512, 1024, 512, 256, 256, 256),
        num_classes=80,
        anchor_generator=dict(
            type='SSDAnchorGenerator',
            scale_major=False,
            input_size=input_size,
            basesize_ratio_range=(0.15, 0.9),
            strides=[8, 16, 32, 64, 100, 300],
            ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2])))
cudnn_benchmark = True
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.5,
        min_pos_iou=0.,
        ignore_iof_thr=-1,
        gt_max_assign_all=False),
    smoothl1_beta=1.,
    allowed_border=-1,
    pos_weight=-1,
    neg_pos_ratio=3,
    debug=False)
test_cfg = dict(
    nms=dict(type='nms', iou_threshold=0.45),
    min_bbox_size=0,
    score_thr=0.02,
    max_per_img=200)
