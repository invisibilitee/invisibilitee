python tools/train.py configs/yolo/yolov3_small.py \
   --gpu-ids 0 \
   --pre_train checkpoints/yolov3_d53_mstrain-608_273e_coco-139f5633.pth \
   --no-validate

python tools/train.py configs/yolo/yolov3_d53_fashion_2.py \
   --gpu-ids 3 \
   --pre_train checkpoints/yolov3_d53_mstrain-608_273e_coco-139f5633.pth \
   --no-validate


# test code

c3 python tools/test.py configs/cornernet/cornernet_hourglass104_mstest_8x6_210e_fashion.py \
    --pre_train ../mmdetection-yax/checkpoints/cornernet_hourglass104_mstest_8x6_210e_coco_20200825_150618-79b44c30.pth \
    --eval bbox


pip install mmcv-full==latest+torch1.7.0+cu92 -f https://download.openmmlab.com/mmcv/dist/index.html


# cornernet
configs/cornernet/cornernet_hourglass104_mstest_8x6_210e_fashion.py
../mmdetection-yax/checkpoints/cornernet_hourglass104_mstest_8x6_210e_coco_20200825_150618-79b44c30.pth


# python draw_pattern.py \
#     --pre_train 模型地址 \
#     –config 训练的时候使用的config地址 \
#     –pattern_name 想要保存的pattern名


# model_path=work_dirs/yolov3_d53_dark2warp_mstrain-608_273e_fashion_randv2/latest.pth
# config_path=configs/yolo/yolov3_d53_dark2warp_mstrain-608_273e_fashion.py




config_path=configs/yolo/yolov3_small.py
model_path=work_dirs/yolov3_small/epoch_2.pth
# pattern_path=p_23.png
pattern_path=patch_imgs/yolov3_small_p_2300.png
# pattern_path=patch_imgs/yolov3_small_p_10900.png

c3 python draw_pattern_new.py \
    --pre_train    $model_path \
    --config       $config_path \
    --pattern_name $pattern_path \
    --result_root  result_yolov3_small_p_2300 \
    --limit 500
    
c3 python draw_infer_imgs.py \
    --pre_train    $model_path \
    --config       $config_path \
    --pattern_name $pattern_path \
    --result_root  result_yolov3_small_p_2300 \
    --limit 500















config_path=configs/yolo/yolov3_small.py
model_path=work_dirs/yolov3_small/epoch_2.pth
# pattern_path=p_23.png
pattern_path=patch_imgs/yolov3_small_p_23100.png
# pattern_path=patch_imgs/yolov3_small_p_10900.png

c3 python draw_pattern_new.py \
    --pre_train    $model_path \
    --config       $config_path \
    --pattern_name $pattern_path \
    --result_root  result_yolov3_small_p_23100 \
    --limit 500
    
c3 python draw_infer_imgs.py \
    --pre_train    $model_path \
    --config       $config_path \
    --pattern_name $pattern_path \
    --result_root  result_yolov3_small_p_23100 \
    --limit 500



















# retinanet
config_path=configs/retinanet/retinanet_r50_fpn_1x_fashion.py
model_path=../mmdetection-yax/checkpoints/retinanet_r50_caffe_fpn_1x_coco_20200531-f11027c5.pth
c3 python tools/test.py \
    $config_path \
    $model_path \
    --eval bbox
    
Evaluating bbox...
Loading and preparing results...
DONE (t=5.49s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=59.34s).
Accumulating evaluation results...
DONE (t=12.46s).
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.252
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.661
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.155
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.000
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.108
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.256
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.468
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.468
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.468
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.200
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.210
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.472
OrderedDict([('bbox_mAP', 0.252), ('bbox_mAP_50', 0.661), ('bbox_mAP_75', 0.155), ('bbox_mAP_s', 0.0), ('bbox_mAP_m', 0.108), ('bb
ox_mAP_l', 0.256), ('bbox_mAP_copypaste', '0.252 0.661 0.155 0.000 0.108 0.256')])






config_path=configs/retinanet/retinanet_r50_fpn_1x_fashion_attack.py
model_path=../mmdetection-yax/checkpoints/retinanet_r50_caffe_fpn_1x_coco_20200531-f11027c5.pth
c3 python tools/test.py \
    $config_path \
    $model_path \
    --eval bbox



config_path=configs/retinanet/retinanet_r50_fpn_1x_fashion_attack2.py
model_path=../mmdetection-yax/checkpoints/retinanet_r50_caffe_fpn_1x_coco_20200531-f11027c5.pth
c2 python tools/test.py \
    $config_path \
    $model_path \
    --eval bbox




pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html













python tools/train.py configs/yolo/yolov3_small_P40.py \
   --gpu-ids 1 \
   --pre_train checkpoints/yolov3_d53_mstrain-608_273e_coco-139f5633.pth \
   --no-validate




python tools/train.py configs/yolo/yolo3_small_all_loss.py \
   --gpu-ids 2 \
   --pre_train checkpoints/yolov3_d53_mstrain-608_273e_coco-139f5633.pth \
   --no-validate



python tools/train.py configs/yolo/yolov3_small_P40_adv.py \
   --gpu-ids 2 \
   --pre_train checkpoints/yolov3_d53_mstrain-608_273e_coco-139f5633.pth \
   --no-validate



python tools/train.py configs/yolo/yolov3_small_P40_lr.py \
   --gpu-ids 0 \
   --pre_train checkpoints/yolov3_d53_mstrain-608_273e_coco-139f5633.pth \
   --no-validate




python tools/train.py configs/yolo/yolov3_small_P40_lr_adv.py \
   --gpu-ids 1 \
   --pre_train checkpoints/yolov3_d53_mstrain-608_273e_coco-139f5633.pth \
   --no-validate



c3 python tools/train.py configs/yolo/yolov3_small_P100.py \
   --gpu-ids 0 \
   --pre_train checkpoints/yolov3_d53_mstrain-608_273e_coco-139f5633.pth \
   --no-validate











config_path=configs/yolo/yolov3_small_P40_adv.py
model_path=work_dirs/yolov3_small_P40_adv/epoch_3.pth
# pattern_path=p_23.png
pattern_path=patch_imgs_adv/p_12200.png
# pattern_path=patch_imgs/yolov3_small_p_10900.png

python draw_pattern_new.py \
    --pre_train    $model_path \
    --config       $config_path \
    --pattern_name $pattern_path \
    --result_root  result_yolov3_p_12200 \
    --limit 500
    
python draw_infer_imgs.py \
    --pre_train    $model_path \
    --config       $config_path \
    --pattern_name $pattern_path \
    --result_root  result_yolov3_p_12200 \
    --limit 500







config_path=configs/yolo/yolov3_small_P40_lr.py
model_path=work_dirs/yolov3_small_P40_adv/epoch_1.pth
# pattern_path=p_23.png
pattern_path=patch_imgs_lr/yolov3_small_p_100.png
# pattern_path=patch_imgs_lr/yolov3_small_p_100.png
# pattern_path=patch_imgs/yolov3_small_p_10900.png

c1 python draw_pattern_new.py \
    --pre_train    $model_path \
    --config       $config_path \
    --pattern_name $pattern_path \
    --result_root  result_yolov3_lr_p_100 \
    --limit 100
    
c1 python draw_infer_imgs.py \
    --pre_train    $model_path \
    --config       $config_path \
    --pattern_name $pattern_path \
    --result_root  result_yolov3_lr_p_100 \
    --limit 100


config_path=configs/yolo/yolov3_small_P40_lr.py
model_path=work_dirs/yolov3_small_P40_adv/epoch_1.pth
# pattern_path=p_23.png
pattern_path=patch_imgs_lr/yolov3_small_p_200.png
# pattern_path=patch_imgs_lr/yolov3_small_p_100.png
# pattern_path=patch_imgs/yolov3_small_p_10900.png

c1 python draw_pattern_new.py \
    --pre_train    $model_path \
    --config       $config_path \
    --pattern_name $pattern_path \
    --result_root  result_yolov3_lr_p_200 \
    --limit 100
    
c1 python draw_infer_imgs.py \
    --pre_train    $model_path \
    --config       $config_path \
    --pattern_name $pattern_path \
    --result_root  result_yolov3_lr_p_200 \
    --limit 100






config_path=configs/yolo/yolov3_small_P40_lr.py
model_path=work_dirs/yolov3_small_P40_adv/epoch_1.pth
pattern_path=patch_imgs_lradv/adv_p_200.png

c2 python draw_pattern_new.py \
    --pre_train    $model_path \
    --config       $config_path \
    --pattern_name $pattern_path \
    --result_root  result_yolov3_adv_p_200 \
    --limit 100
    
c2 python draw_infer_imgs.py \
    --pre_train    $model_path \
    --config       $config_path \
    --pattern_name $pattern_path \
    --result_root  result_yolov3_adv_p_200 \
    --limit 100







config_path=configs/yolo/yolov3_small_P40_lr.py
model_path=work_dirs/yolov3_small_P40_adv/epoch_1.pth
pattern_path=patch_imgs_lradv/adv_p_9000.png

c2 python draw_pattern_new.py \
    --pre_train    $model_path \
    --config       $config_path \
    --pattern_name $pattern_path \
    --result_root  result_yolov3_adv_p_9000 \
    --limit 100
    
c2 python draw_infer_imgs.py \
    --pre_train    $model_path \
    --config       $config_path \
    --pattern_name $pattern_path \
    --result_root  result_yolov3_adv_p_9000 \
    --limit 100








config_path=configs/yolo/yolov3_small_P40_lr.py
model_path=work_dirs/yolov3_small_P40_adv/epoch_1.pth
pattern_path=patch_imgs/yolov3_small_p_62700.png

c2 python draw_pattern_new.py \
    --pre_train    $model_path \
    --config       $config_path \
    --pattern_name $pattern_path \
    --result_root  result_yolov3_small_p_62700 \
    --limit 100
    
c2 python draw_infer_imgs.py \
    --pre_train    $model_path \
    --config       $config_path \
    --pattern_name $pattern_path \
    --result_root  result_yolov3_small_p_62700 \
    --limit 100









config_path=configs/yolo/yolov3_small_P100.py
model_path=work_dirs/yolov3_small/epoch_1.pth
pattern_path=patch_imgs_p100/p_28100.png


c2 python draw_pattern_new.py \
    --pre_train    $model_path \
    --config       $config_path \
    --pattern_name $pattern_path \
    --result_root  result_yolov3_small_p100_28100 \
    --limit 50
    
c2 python draw_infer_imgs.py \
    --pre_train    $model_path \
    --config       $config_path \
    --pattern_name $pattern_path \
    --result_root  result_yolov3_small_p100_28100 \
    --limit 50






config_path=configs/yolo/yolov3_small_P40_lr.py
model_path=work_dirs/yolov3_small_P40_adv/epoch_1.pth
pattern_path=patch_imgs_lr/yolov3_small_p_14000.png

c2 python draw_pattern_new.py \
    --pre_train    $model_path \
    --config       $config_path \
    --pattern_name $pattern_path \
    --result_root  result_yolov3_small_p_14000 \
    --limit 100
    
c2 python draw_infer_imgs.py \
    --pre_train    $model_path \
    --config       $config_path \
    --pattern_name $pattern_path \
    --result_root  result_yolov3_small_p_14000 \
    --limit 100




config_path=configs/yolo/yolov3_small_P40_lr.py
model_path=work_dirs/yolov3_small_P40_adv/epoch_1.pth
pattern_path=patch_imgs_lr_gt/yolov3_small_p_5900.png

c3 python draw_pattern_new.py \
    --pre_train    $model_path \
    --config       $config_path \
    --pattern_name $pattern_path \
    --result_root  result_yolov3_small_lrgt_5900 \
    --limit 100
    
c3 python draw_infer_imgs.py \
    --pre_train    $model_path \
    --config       $config_path \
    --pattern_name $pattern_path \
    --result_root  result_yolov3_small_lrgt_5900 \
    --limit 100


















c3 python tools/test.py configs/cornernet/cornernet_small_fashion_1.py \
    ../mmdetection-yax/checkpoints/cornernet_hourglass104_mstest_8x6_210e_coco_20200825_150618-79b44c30.pth \
    --eval bbox












# 2021年05月12日

#95
python tools/train.py configs/yolo/yolov3_downsample_p100.py \
   --gpu-ids 0 \
   --pre_train checkpoints/yolov3_d53_mstrain-608_273e_coco-139f5633.pth \
   --no-validate

#95
python tools/train.py configs/yolo/yolov3_downsample_p200.py \
   --gpu-ids 1 \
   --pre_train checkpoints/yolov3_d53_mstrain-608_273e_coco-139f5633.pth \
   --no-validate

#114
python tools/train.py configs/yolo/yolov3_downsample_300p100.py \
   --gpu-ids 0 \
   --pre_train checkpoints/yolov3_d53_mstrain-608_273e_coco-139f5633.pth \
   --no-validate


#114
python tools/train.py configs/yolo/yolov3_downsample_224p100_clsloss.py \
   --gpu-ids 3 \
   --pre_train checkpoints/yolov3_d53_mstrain-608_273e_coco-139f5633.pth \
   --no-validate


# test

config_path=configs/yolo/yolo3_down_test.py
model_path=work_dirs/yolov3_downsample_p100/latest.pth
pattern_path=patch_downsample_p100/p_15300.png
result_folder=result_yolov3_down_p100_15300

python draw_pattern_new.py \
    --pre_train    work_dirs/yolov3_downsample_p100/latest.pth \
    --config       configs/yolo/yolo3_down_test.py \
    --pattern_name $pattern_path \
    --result_root  $result_folder \
    --limit 50
    
python draw_infer_imgs.py \
    --pre_train    $model_path \
    --config       $config_path \
    --pattern_name $pattern_path \
    --result_root  $result_folder \
    --limit 50




config_path=configs/yolo/yolo3_down_test.py
model_path=work_dirs/yolov3_downsample_p100/latest.pth
pattern_path=patch_downsample_p200/p_15300.png
result_folder=result_yolov3_down_p200_15300

python draw_pattern_new.py \
    --pre_train    work_dirs/yolov3_downsample_p100/latest.pth \
    --config       configs/yolo/yolo3_down_test.py \
    --pattern_name $pattern_path \
    --result_root  $result_folder \
    --limit 50
    
python draw_infer_imgs.py \
    --pre_train    $model_path \
    --config       $config_path \
    --pattern_name $pattern_path \
    --result_root  $result_folder \
    --limit 50







config_path=configs/yolo/yolo3_down_test.py
model_path=work_dirs/yolov3_downsample_p100/latest.pth
pattern_path=patch_downsample_p200/p_15300.png
result_folder=result_yolov3_down_300p100

python draw_pattern_new.py \
    --pre_train    work_dirs/yolov3_downsample_p100/latest.pth \
    --config       configs/yolo/yolo3_down_test.py \
    --pattern_name $pattern_path \
    --result_root  $result_folder \
    --limit 50
    
python draw_infer_imgs.py \
    --pre_train    $model_path \
    --config       $config_path \
    --pattern_name $pattern_path \
    --result_root  $result_folder \
    --limit 50





config_path=configs/yolo/yolo3_down_test.py
model_path=work_dirs/yolov3_downsample_p100/latest.pth
pattern_path=patch_downsample_224p100/p_2100.png
result_folder=result_yolov3_down_224p100

c2 python draw_pattern_new.py \
    --pre_train    work_dirs/yolov3_downsample_p100/latest.pth \
    --config       configs/yolo/yolo3_down_test.py \
    --pattern_name $pattern_path \
    --result_root  $result_folder \
    --limit 100
    
c2 python draw_infer_imgs.py \
    --pre_train    $model_path \
    --config       $config_path \
    --pattern_name $pattern_path \
    --result_root  $result_folder \
    --limit 100



