from mmdet.apis import init_detector, inference_detector
import mmcv
import os
from tqdm import tqdm
import json

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

def person_detection(model, img):
    flag = False
    result = inference_detector(model, img)
    if isinstance(result, tuple):
        bbox_result, segm_result = result
        if isinstance(segm_result, tuple):
            segm_result = segm_result[0]  # ms rcnn
    else:
        bbox_result, segm_result = result, None
    if not len(bbox_result[0]) == 0: # person in picture
        flag = True
    return flag

root_mm = '/home/yaxian_li/expriment/mmdetection-yax'
# Specify the path to model config and checkpoint file
#config_file = os.path.join(root_mm, 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py')
#checkpoint_file = os.path.join(root_mm,'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth')

config_file = os.path.join(root_mm, 'configs/cocornernet_hourglass104_mstest_8x6_210e_coco.py')
checkpoint_file = os.path.join(root_mm,'checkpoints/cornernet_hourglass104_mstest_8x6_210e_coco_20200825_150618-79b44c30.pth')

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results

img_path = 'new_pic'
img_file_list = os.listdir(img_path)

#file = 'val_fashion_faster_rcnn_r50_fpn_1x.txt' # detection result, list of imags with person
person_in_img = []
cnt_cate = [[0, 0, 0] for i in range(15)]

for img_name in tqdm(img_file_list):
    # or img = mmcv.imread(img), which will only load it once
    img_id = img_name.split('.')[0]
    img = os.path.join(img_path, img_name)
    result = inference_detector(model, img)
    # flag = person_detection(model, img)
    # or save the visualization results to image files
    target = os.path.join('case', '{}_bbox.jpg'.format(img_id))
    model.show_result(img, result, out_file=target)


'''
with open(file, mode='w+') as f:
    tmp = '\n'.join(i for i in person_in_img)
    f.write(tmp) 
f.close()

# test a video and show the results

video = mmcv.VideoReader('video.mp4')
for frame in video:
    result = inference_detector(model, frame)
    model.show_result(frame, result, wait_time=1)
'''