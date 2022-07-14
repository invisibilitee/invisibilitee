_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn_attack.py',
    '../_base_/datasets/fashion_detection.py', #yaxian
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

test_cfg = dict(
    rcnn=dict(
        score_thr=0.05,
        nms=dict(type='soft_nms', iou_threshold=0.5),
        max_per_img=100))
