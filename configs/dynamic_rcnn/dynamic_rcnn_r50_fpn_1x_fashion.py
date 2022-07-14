_base_ = '../faster_rcnn/faster_rcnn_r50_fpn_1x_fashion.py'
patch_saved_path = '/home/yaxian_li/expriment/mmdetection-yax/patch_downsample_300p100/p_150400.png'

model = dict(
    backbone=dict(
        patch_path=patch_saved_path, #yaxian
        training=False
    ),
    roi_head=dict(
        type='DynamicRoIHead',
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=80,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))))
train_cfg = dict(
    rpn_proposal=dict(nms_thr=0.85),
    rcnn=dict(
        dynamic_rcnn=dict(
            iou_topk=75,
            beta_topk=10,
            update_iter_interval=100,
            initial_iou=0.4,
            initial_beta=1.0)))
test_cfg = dict(rpn=dict(nms_thr=0.85))