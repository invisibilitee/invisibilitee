# draw gt_bbox.py
import json
import os
import cv2

data_root = '/DATASETS_2/DeepFashion2/'
dataset_type = 'DeepFashionDataset'
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annos_attack/deepfashion2_human_train.json',
        img_prefix=data_root + 'train/image/'),
    val=dict(
        type=dataset_type,
        #samples_per_gpu=2,
        ann_file=data_root + 'annos_attack/deepfashion2_human_val.json',
        img_prefix=data_root + 'validation/image/'),
    test=dict(
        #samples_per_gpu=16,
        type=dataset_type,
        #ann_file=data_root + 'annos_attack/deepfashion2_physical_demo2.json', #yaxian
        #img_prefix='/home/yaxian_li/expriment/FashionAI_KeyPoint_Detection_Challenge_Keras-master/outputs/demo2',
        ann_file=data_root +'annos_attack/deepfashion2_human_val.json',
        img_prefix=data_root + 'validation/image/',
        #img_prefix=data_root + 'annos_attack/test2/'
        )
)

colors = (0,0,255)
save_path = data_root + 'annos_attack/gt_bbox/'
if not os.path.exists(save_path):
    os.mkdir(save_path)

json_name = data['test']['ann_file']

with open(json_name) as f:
    test_dataset = json.load(f)
    for i, img_info in enumerate(test_dataset['images']):
        ann_info = test_dataset['annotations'][i]
        bbox = ann_info['bbox']
        img_name = img_info['file_name']
        img_path = os.path.join(data['test']['img_prefix'], img_name)
        img = cv2.imread(img_path)
        x,y,w,h = bbox
        cv2.rectangle(img, (x, y), (x+w,y+h), colors, 5)
        cv2.imwrite(os.path.join(save_path, img_name), img)
