_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py', # yaxian
    #'../_base_/models/mask_rcnn_r50_fpn_attack.py', # yaxian
    '../_base_/datasets/fashion_detection.py', #yaxian
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
