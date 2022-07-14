_base_ = '../_base_/default_runtime.py'
# 训练时 patch原始尺寸：参数数目=patch_ori_size * patch_ori_size * 3
patch_ori_size = 224

# 训练时 imgresize 大小
img_resize = 224

# 训练/测试 时patch目标尺寸：将patch downsample到目标尺寸后加入原图
patch_size=100

# 训练时 batch_size
samples_per_gpu=8

# 训练时 中间patch存储目录
patch_patch = '/home/yaxian_li/expriment/mmdetection-yax/patch_downsample_p100/p_9900.png'

# 使用darknet3：最新版本
model = dict(
    type='YOLOV3',
    #pretrained='open-mmlab://darknet53',
    #pretrained="./checkpoints/yolov3_d53_mstrain-608_273e_coco-139f5633.pth",
    #pretrained='work_dirs/yolov3_d53_dark2warp_mstrain-608_273e_fashion/epoch_15.pth',
    backbone=dict(type='Darknet3', depth=53, out_indices=(3, 4, 5),
        patch_path      = patch_patch,
        patch_init_path = patch_patch,
        patch_size      = patch_size,
        patch_ori_size  = patch_ori_size,
        training=False #yaxian
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
            use_sigmoid=True,
            negative=True,
            loss_weight=1.0,
            reduction='sum'),

        loss_xy=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            negative=True,
            loss_weight=0.0,
            reduction='sum'),

        loss_wh=dict(type='MSELoss', 
            loss_weight=0.0, reduction='sum', negative=True),

        loss_tv=dict(type='TVLoss', loss_weight=100),
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
        img_scale=(1333, 800),
        # flip=False,
        transforms=[
            dict(type='RandomFlip'),
            dict(type='Resize', keep_ratio=True),
            # dict(type='Resize', img_scale=[(512, 512), (512, 512)], keep_ratio=True),
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
optimizer = dict(type='Adam', lr=0.1)

#optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=100,  # same as burn-in in darknet
    warmup_ratio=0.01, # yaxian changed
    step=[218, 246])
# runtime settings
total_epochs = 200
evaluation = dict(interval=1, metric=['bbox'])


log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])