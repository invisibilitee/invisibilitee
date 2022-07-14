_base_ = [
    '../_base_/default_runtime.py',
    #'../_base_/datasets/fashion_detection.py' #yaxian
]
patch_saved_path = '/home/yaxian_li/expriment/mmdetection-yax/faster_rand_p100/faster_rand_p50_123800_down.png'
# model settings
model = dict(
    type='CornerNet', 
    backbone=dict(
        type='HourglassNet2', # yaxian
        downsample_times=5,
        num_stacks=2,
        stage_channels=[256, 256, 384, 384, 384, 512],
        stage_blocks=[2, 2, 2, 2, 2, 4],
        norm_cfg=dict(type='BN', requires_grad=True),
        patch_path=patch_saved_path, #yaxian
        training=False), #yaxian
    neck=None,
    bbox_head=dict(
        type='CornerHead',
        num_classes=80,
        in_channels=256,
        num_feat_levels=2,
        corner_emb_channels=1,
        loss_heatmap=dict(
            type='GaussianFocalLoss', alpha=2.0, gamma=4.0, loss_weight=1),
        loss_embedding=dict(
            type='AssociativeEmbeddingLoss',
            pull_weight=0.10,
            push_weight=0.10),
        loss_offset=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1)))
# data settings, yaxian
#img_norm_cfg = dict(
#    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
img_norm_cfg = dict(mean=[0, 0, 0], std=[255., 255., 255.], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(
        type='RandomCenterCropPad',
        crop_size=(511, 511),
        ratios=(0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3),
        test_mode=False,
        test_pad_mode=None,
        **img_norm_cfg),
    dict(type='Resize', img_scale=(511, 511), keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_keypoints']), #yaxian
]
test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(
        type='MultiScaleFlipAug',
        scale_factor=1.0,
        #img_scale=(416, 416),
        flip=True,
        transforms=[
            dict(type='Resize', img_scale=(900, 900), keep_ratio=True),
            
            dict(
                type='RandomCenterCropPad',
                crop_size=None,
                ratios=None,
                border=None,
                test_mode=True,
                test_pad_mode=['logical_or', 127],
                **img_norm_cfg),
            
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),#yaxian
            dict(type='ImageToTensor', keys=['img']), 
            dict(
                type='Collect',
                keys=['img', 'gt_keypoints'], # yaxian
                meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape',
                           'scale_factor', 'flip', 'img_norm_cfg', 'border')),
        ])
]

dataset_type = 'DeepFashionDataset'
data_root = '/DATASETS_2/DeepFashion2/' #yaxian
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annos_attack/deepfashion2_human_train.json',
        img_prefix=data_root + 'train/image/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        #samples_per_gpu=2,
        ann_file=data_root + 'annos_attack/deepfashion2_human_val.json',
        img_prefix=data_root + 'validation/image/',
        pipeline=test_pipeline),
    test=dict(
        #samples_per_gpu=1,
        type=dataset_type,
        #ann_file=data_root + 'annos_attack/deepfashion2_physical_demo4.json', #yaxian
        #img_prefix='/home/yaxian_li/expriment/FashionAI_KeyPoint_Detection_Challenge_Keras-master/outputs/demo4',
        ann_file=data_root +'annos_attack/deepfashion2_human_val.json',
        img_prefix=data_root + 'validation/image/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')

# training and testing settings
train_cfg = None
test_cfg = dict(
    corner_topk=100,
    local_maximum_kernel=3,
    distance_threshold=0.5,
    score_thr=0.05,
    max_per_img=100,
    nms_cfg=dict(type='soft_nms', iou_threshold=0.5, method='gaussian'))
# optimizer
optimizer = dict(type='Adam', lr=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[180])
total_epochs = 210
