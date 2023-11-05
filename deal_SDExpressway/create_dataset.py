# python json_to_dataset.py

import argparse
import json
import os
import os.path as osp
import warnings
import copy
import numpy as np
import PIL.Image
from skimage import io
import yaml
from labelme import utils
from PIL import Image
import cv2

def save_json(save_path,data):
    assert save_path.split('.')[-1] == 'json'
    with open(save_path,'w') as file:
        json.dump(data,file,indent=4)
 
def load_json(file_path):
    assert file_path.split('.')[-1] == 'json'
    with open(file_path,'r') as file:
        data = json.load(file)
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=False, default='images')   
    parser.add_argument('--drivable_area', required=False, default='drivable area labels_sub_png')  
    parser.add_argument('--lane_line', required=False, default='lane line labels_png')  
    parser.add_argument('--traffic_object', required=False, default='traffic object labels_json')   
    parser.add_argument('--dataset', required=False, default='SDExpressway')   
    args = parser.parse_args()
    image = args.image
    drivable_area = args.drivable_area
    lane_line = args.lane_line
    traffic_object = args.traffic_object
    dataset = args.dataset

    list = os.listdir(image)   # 获取image文件列表
    for i in range(1, len(list)+1):
        print("NO", i)
        img_pth = os.path.join(image, '{}.jpg'.format(i))
        dri_pth = os.path.join(drivable_area, '{}.png'.format(i))
        lan_pth = os.path.join(lane_line, '{}.png'.format(i))
        tra_pth = os.path.join(traffic_object, '{}.json'.format(i))

        img = Image.open(img_pth)
        dri = Image.open(dri_pth)
        lan = Image.open(lan_pth)
        tra = load_json(tra_pth)
        if i%5 == 0:
            img_name = os.path.join('SDExpressway/images/val', '{}.jpg'.format(i))
            dri_name = os.path.join('SDExpressway/drivable area labels/val', '{}.png'.format(i))
            lan_name = os.path.join('SDExpressway/lane line labels/val', '{}.png'.format(i))
            tra_name = os.path.join('SDExpressway/traffic object labels/val', '{}.json'.format(i))
            img.save(img_name)
            # dri.save(dri_name)
            # lan.save(lan_name)
            # save_json(tra_name,tra)
        else:
            img_name = os.path.join('SDExpressway/images/train', '{}.jpg'.format(i))
            dri_name = os.path.join('SDExpressway/drivable area labels/train', '{}.png'.format(i))
            lan_name = os.path.join('SDExpressway/lane line labels/train', '{}.png'.format(i))
            tra_name = os.path.join('SDExpressway/traffic object labels/train', '{}.json'.format(i))
            img.save(img_name)
            # dri.save(dri_name)
            # lan.save(lan_name)
            # save_json(tra_name,tra)
        
if __name__ == '__main__':
    main()