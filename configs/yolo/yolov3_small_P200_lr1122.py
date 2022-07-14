_base_ = '../_base_/default_runtime.py'
# attack settings
# model settings
patch_size=200
patch_ori_size=400
patch_init_path = '/home/yaxian_li/expriment/mmdetection-yax/patch_imgs_200_lr/yolov3_p200_100.png'
patch_saved_path = '/home/yaxian_li/expriment/mmdetection-yax/patch_lr-3_1122/yolov3_1122_try.png'
model = dict(
    type='YOLOV3',
    backbone=dict(type='Darknet3', depth=53, out_indices=(3, 4, 5),
        patch_ori_size=patch_ori_size,
        patch_path=patch_saved_path, #yaxian
        patch_size=patch_size,
        patch_init_path=patch_init_path,
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
            loss_weight=5.0,
            reduction='sum'),
        
        loss_conf=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            negative=True,
            loss_weight=1.0,
            reduction='sum'),

        loss_xy=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            negative=True,
            loss_weight=1.0,
            reduction='sum'),

        loss_wh=dict(type='MSELoss', 
            loss_weight=1.0, reduction='sum', negative=True),

        loss_tv=dict(type='TVLoss', loss_weight=100),
        loss_nps=dict(type='NPSLoss', patch_size=patch_ori_size, loss_weight=100)))
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
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Expand',
        mean=img_norm_cfg['mean'],
        to_rgb=img_norm_cfg['to_rgb'],
        ratio_range=(1, 2)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        min_crop_size=0.3),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Resize', img_scale=[(224, 224), (224, 224)], keep_ratio=False),
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
samples_per_gpu=40
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
optimizer = dict(type='Adam', lr=0.001)

#optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=None)



# 5.lr_scheduler.ReduceLROnPlateau

# class torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
# mode='min', factor=0.1, patience=10, 
# verbose=False, threshold=0.0001, 
# threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

# learning policy
lr_config = dict(
    policy='step',
    step=2500,
    gamma=0.1,
    )
# runtime settings
total_epochs = 200
evaluation = dict(interval=1, metric=['bbox'])


log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])