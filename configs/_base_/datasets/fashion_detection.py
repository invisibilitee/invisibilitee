dataset_type = 'DeepFashionDataset'
data_root = '/DATASETS_2/DeepFashion2/' #yaxian
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(416, 416), keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_keypoints']), #yaxian
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(416, 416),
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=(416, 416), keep_ratio=False),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']), #yaxian , 'gt_keypoints'
        ])
]

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
        #samples_per_gpu=16,
        type=dataset_type,
        #ann_file=data_root + 'annos_attack/deepfashion2_physical_demo2.json', #yaxian
        #img_prefix='/home/yaxian_li/expriment/FashionAI_KeyPoint_Detection_Challenge_Keras-master/outputs/demo2',
        ann_file=data_root +'annos_attack/deepfashion2_human_val.json',
        img_prefix=data_root + 'validation/image/',
        #img_prefix=data_root + 'annos_attack/test_yolo2/',
        #img_prefix=data_root + 'annos_attack/test2/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
