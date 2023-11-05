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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--drivable_area_file', required=False, default='labels_drivable_area_sub')  
    parser.add_argument('--lane_line_file', required=False, default='labels_lane_line_color')   
    parser.add_argument('--traffic_object_file', required=False, default='labels_traffic_object_fusion')   
    parser.add_argument('--all_fusion_file', required=False, default='images_all_fusion')   
    args = parser.parse_args()

    drivable_area_file = args.drivable_area_file
    lane_line_file = args.lane_line_file
    traffic_object_file = args.traffic_object_file
    all_fusion_file = args.all_fusion_file

    list = os.listdir(drivable_area_file)   # 获取 文件列表
    for i in range(0, len(list)):
        filename = list[i][:-4]       # 提取出.png前的字符作为文件名，以便后续使用
        drivable_area_path = drivable_area_file+'/'+'{}.png'.format(filename)
        lane_line_path = lane_line_file+'/'+'{}.png'.format(filename)
        traffic_object_path = traffic_object_file+'/'+'{}.png'.format(filename)
        all_fusion_path = all_fusion_file+'/'+'{}.png'.format(filename)


        # image = np.ascontiguousarray(image)
        img1 = cv2.imread(drivable_area_path, cv2.IMREAD_COLOR)
        img2 = cv2.imread(lane_line_path, cv2.IMREAD_COLOR)
        img3 = cv2.imread(traffic_object_path, cv2.IMREAD_COLOR)
        combine1 = cv2.addWeighted(img3, 1, img1,0.3, 0)
        combine2 = cv2.addWeighted(combine1, 1, img2,0.3, 0)
        cv2.imwrite(all_fusion_path,combine2,[int(cv2.IMWRITE_PNG_COMPRESSION),9])
        print('Saved to: %s' % all_fusion_file)


if __name__ == '__main__':
    main()