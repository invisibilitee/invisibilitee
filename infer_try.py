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
config_file = os.path.join(root_mm, 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py')
checkpoint_file = os.path.join(root_mm,'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth')

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
datapath = '/DATASETS_2/DeepFashion2'
whole_path = {
    'train': os.path.join(datapath, 'train', 'annos'),
    'val':os.path.join(datapath, 'validation', 'annos'),
    'train_img': os.path.join(datapath, 'train', 'image'),
    'val_img':os.path.join(datapath, 'validation', 'image'),
    'test':os.path.join(datapath, 'test')
}

json_path = whole_path['val']
img_path = whole_path['val_img']
img_file_list = os.listdir(img_path)

#file = 'val_fashion_faster_rcnn_r50_fpn_1x.txt' # detection result, list of imags with person
person_in_img = []
cnt_cate = [[0, 0, 0] for i in range(15)]

for img_name in tqdm(img_file_list):
    # or img = mmcv.imread(img), which will only load it once
    flag = False
    img = os.path.join(img_path, img_name)
    json_file = os.path.join(json_path, img_name.split('.')[0]+'.json')
    if not (os.path.exists(json_file)):
        print('{} do not exsit!'.format(json_file))
        continue
    with open(json_file, encoding='utf-8') as f:
        label = json.load(f)
        for key, val in label.items():
            # key: 'item1','item2' 'source' 'pair_id'
            # val: 'bounding_box'  'category_name' 'scale' 'viewpoint' 'occlusion'....
            if 'item' in key:
                if label[key]['category_id'] == 1:
                    points_x = label[key]['landmarks'][0::3]
                    points_y = label[key]['landmarks'][1::3]
                    points_v = label[key]['landmarks'][2::3]
                    for i in range(len(points_x)):
                        print(points_x[i], points_y[i], points_v[i])
                    flag = person_detection(model, img, label[key]['keypoints'])
                    break;
                    if flag:
                        cnt_cate[label[key]['category_id']][label[key]['viewpoint']-1] += 1
        ''''
        if not label[key]['viewpoint'] == 1:
            person_in_img.append(img_name)
        '''
    f.close()
    break
    
    # or save the visualization results to image files
    #target = os.path.join(datapath, 'tmp', 'result.jpg')
    model.show_result(img, result, out_file=target)
    #break

#print('#images {}'.format(cnt))
for i in cnt_cate:
    print('{}\t{}\t{}'.format(i[0],i[1],i[2]))
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