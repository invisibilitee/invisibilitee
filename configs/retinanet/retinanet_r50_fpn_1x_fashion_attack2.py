_base_ = [
    # '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/models/retinanet_r50_fpn_attack2.py',
    # '../_base_/models/retinanet_r50_fpn_attack.py',

    '../_base_/datasets/fashion_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]


img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            # dict(type='Collect', keys=['img']),
            dict(type='Collect', keys=['img', 'gt_keypoints'])

        ])
]

dataset_type = 'DeepFashionDataset' # change into according dataset
data_root = '/DATASETS_2/DeepFashion2/'
samples_per_gpu=16
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
        #samples_per_gpu=4,
        type=dataset_type,
        ann_file=data_root + 'annos_attack/deepfashion2_human_val.json',
        img_prefix=data_root + 'validation/image/',
        pipeline=test_pipeline))