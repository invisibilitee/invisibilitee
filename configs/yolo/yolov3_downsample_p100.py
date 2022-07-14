_base_ = '../_base_/default_runtime.py'
# attack settings
# model settings

# 训练时 patch原始尺寸：参数数目=patch_ori_size * patch_ori_size * 3
patch_ori_size = 416

# 训练时 imgresize 大小
img_resize = 416

# 训练/测试 时patch目标尺寸：将patch downsample到目标尺寸后加入原图
patch_size=100

# 训练时 batch_size
samples_per_gpu=16

# 训练时 中间patch存储目录
patch_saved_path = '/home/yaxian_li/expriment/mmdetection-yax/patch_downsample_p100/p.png'

# 测试时 load的patch图像
patch_init_path = ''

# 使用darknet3：最新版本
model = dict(
    type='YOLOV3',
    #pretrained='open-mmlab://darknet53',
    #pretrained="./checkpoints/yolov3_d53_mstrain-608_273e_coco-139f5633.pth",
    #pretrained='work_dirs/yolov3_d53_dark2warp_mstrain-608_273e_fashion/epoch_15.pth',
    backbone=dict(type='Darknet3', depth=53, out_indices=(3, 4, 5),
        patch_path      = patch_saved_path,
        patch_init_path = patch_init_path,
        patch_size      = patch_size,
        patch_ori_size  = patch_ori_size,
        #training=False #yaxian
    ),
    neck=dict(
        type='YOLOV3Neck',
        num_scales=3,
        in_channels=[1024, 512, 256],
        out_channels=[512, 256, 128]),
    bbox_head=dict(
        type='YOLOV3Head',
        num_classes=80, #change according to the dataset
        in_channels=[512, 256, 128],
        out_channels=[1024, 512, 256],
        anchor_generator=dict(
            type='YOLOAnchorGenerator',
            base_sizes=[[(116, 90), (156, 198), (373, 326)],
                        [(30, 61), (62, 45), (59, 119)],
                        [(10, 13), (16, 30), (33, 23)]],
            strides=[32, 16, 8]),
        bbox_coder=dict(type='YOLOBBoxCoder'),
        featmap_strides=[32, 16, 8],
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            negative=True,
            loss_weight=0.0,
            reduction='sum'),
        
        loss_conf=dict(
            type='CrossEntropyLoss',
            use_adversarial=True,
            loss_weight=10000.0,
            reduction='sum'),

        loss_xy=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            negative=True,
            loss_weight=0.0,
            reduction='sum'),

        loss_wh=dict(type='MSELoss', 
            loss_weight=0.0, reduction='sum', negative=True),

        loss_tv=dict(type='TVLoss', loss_weight=10),
        loss_nps=dict(type='NPSLoss',patch_size=patch_ori_size, loss_weight=100)))
# training and testing settings
train_cfg = dict(
    assigner=dict(
        type='GridAssigner', pos_iou_thr=0.5, neg_iou_thr=0.5, min_pos_iou=0))
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    conf_thr=0.005,
    nms=dict(type='nms', iou_threshold=0.45),
    max_per_img=100)


# dataset settings

img_norm_cfg = dict(mean=[0, 0, 0], std=[255., 255., 255.], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),


    # Apply photometric distortion to image sequentially, every transformation
    # is applied with a probability of 0.5. The position of random contrast is in
    # second or second to last.

    # 1. random brightness
    # 2. random contrast (mode 0)
    # 3. convert color from BGR to HSV
    # 4. random saturation
    # 5. random hue
    # 6. convert color from HSV to BGR
    # 7. random contrast (mode 1)
    # 8. randomly swap channels

    # Args:
    #     brightness_delta (int): delta of brightness.
    #     contrast_range (tuple): range of contrast.
    #     saturation_range (tuple): range of saturation.
    #     hue_delta (int): delta of hue.
    # dict(type='PhotoMetricDistortion'),

    # Random expand the image & bboxes.

    # Randomly place the original image on a canvas of 'ratio' x original image
    # size filled with mean values. The ratio is in the range of ratio_range.

    # Args:
    #     mean (tuple): mean value of dataset.
    #     to_rgb (bool): if need to convert the order of mean to align with RGB.
    #     ratio_range (tuple): range of expand ratio.
    #     prob (float): probability of applying this transformation
    # dict(
    #     type='Expand',
    #     mean=img_norm_cfg['mean'],
    #     to_rgb=img_norm_cfg['to_rgb'],
    #     ratio_range=(1, 2)),

    #Random crop the image & bboxes, the cropped patches have minimum IoU
    #requirement with original image & bboxes, the IoU threshold is randomly
    #selected from min_ious.
    # dict(
    #     type='MinIoURandomCrop',
    #     min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    #     min_crop_size=0.3),

    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Resize', img_scale=[(img_resize, img_resize), (img_resize, img_resize)], keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_keypoints']) #yaxian
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 512),
        # flip=False,
        transforms=[
            dict(type='RandomFlip'),
            dict(type='Resize', keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img', 'gt_keypoints']) #yaxian
        ])
]

dataset_type = 'DeepFashionDataset' # change into according dataset
data_root = '/DATASETS_2/DeepFashion2/'

data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annos_attack/deepfashion2_human_train.json',
        img_prefix=data_root + 'train/image/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        samples_per_gpu=samples_per_gpu,
        ann_file=data_root + 'annos_attack/deepfashion2_human_val.json',
        img_prefix=data_root + 'validation/image/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annos_attack/deepfashion2_human_val.json',
        img_prefix=data_root + 'validation/image/',
        pipeline=test_pipeline))



# optimizer
optimizer       = dict(type='Adam', lr=0.01)
optimizer_config = dict(grad_clip=None)



# 5.lr_scheduler.ReduceLROnPlateau

# class torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
# mode='min', factor=0.1, patience=10, 
# verbose=False, threshold=0.0001, 
# threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

# learning policy
lr_config = dict(
    policy='step',
    step=10000,
    gamma=0.5,
    )
# runtime settings
total_epochs = 100
evaluation = dict(interval=1, metric=['bbox'])


log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])