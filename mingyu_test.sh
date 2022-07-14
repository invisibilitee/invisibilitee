


# ssd for test

# 修改这个
/home/yaxian_li/expriment/mmdetection-yax/configs/_base_/models/ssd300_attack_downsample.py

# 这个不动，都使用这个config
/home/yaxian_li/expriment/mmdetection-yax/configs/ssd/ssd300_fashion_downsample.py

config_path=configs/ssd/ssd300_fashion_downsample.py
model_path=../mmdetection-yax/checkpoints/ssd300_coco_20200307-a92d2092.pth
c2 python tools/test.py \
    $config_path \
    $model_path \
    --eval bbox

Use load_from_local loader
[                                                  ] 0/5586, elapsed: 0s, ETA:/home/yaxian_li/expriment/mmdetection-yax/mmdet/models/detectors/single_stage.py:213: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
  gt_keypoints = torch.tensor(gt_keypoints[0])
/home/mingyu_zhang/anaconda3/lib/python3.6/site-packages/torch/nn/functional.py:3385: UserWarning: Default grid_sample and affine_grid behavior has changed to align_corners=False since 1.3.0. Please specify align_corners=True if the old behavior is desired. See the documentation of grid_sample for details.
  warnings.warn("Default grid_sample and affine_grid behavior has changed "
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 5586/5586, 12.6 task/s, elapsed: 444s, ETA:     0s
Evaluating bbox...
Loading and preparing results...
DONE (t=9.76s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=24.30s).
Accumulating evaluation results...
DONE (t=9.04s).
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.000
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.000
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.000
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.000
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.000
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.000
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.000
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.000
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.000
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.000
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.000
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.000
OrderedDict([('bbox_mAP', 0.0), ('bbox_mAP_50', 0.0), ('bbox_mAP_75', 0.0), ('bbox_mAP_s', 0.0), ('bbox_mAP_m', 0.0), ('bbox_mAP_l', 0.0), ('bbox_mAP_copypaste', '0.000 0.000 0.000 0.000 0.000 0.000')])










config_path=configs/yolo/yolo3_down_test.py
model_path=../mmdetection-yax/checkpoints/yolov3_d53_mstrain-608_273e_coco-139f5633.pth
c2 python tools/test.py \
    $config_path \
    $model_path \
    --eval bbox
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 5586/5586, 8.2 task/s, elapsed: 684s, ETA:     0s
Evaluating bbox...
Loading and preparing results...
DONE (t=0.96s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=16.04s).
Accumulating evaluation results...
DONE (t=3.09s).
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.003
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.022
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.000
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.000
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.009
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.003
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.039
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.039
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.039
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.000
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.038
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.039





config_path=configs/cornernet/cornernet_small_fashion_downsample.py
model_path=../mmdetection-yax/checkpoints/cornernet_hourglass104_mstest_8x6_210e_coco_20200825_150618-79b44c30.pth
c2 python tools/test.py \
    $config_path \
    $model_path \
    --eval bbox
DONE (t=10.28s).
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.168
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.466
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.095
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.000
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.087
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.176
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.560
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.560
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.560
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.000
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.324
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.563
OrderedDict([('bbox_mAP', 0.168), ('bbox_mAP_50', 0.466), ('bbox_mAP_75', 0.095), ('bbox_mAP_s', 0.0), ('bbox_mAP_m', 0.087), ('bbox_mAP_l', 0.176), ('bbox_mAP_copypaste', '0.168 0.466 0.095 0.000 0.087 0.176')])