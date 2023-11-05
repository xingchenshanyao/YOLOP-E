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
    parser.add_argument('--image_file', required=False, default='images')   # 原图image所在的文件夹
    parser.add_argument('--drivable_area_file', required=False, default='labels_drivable_area_color')   # 可驾驶区域的掩膜png所在的文件夹
    parser.add_argument('--lane_line_file', required=False, default='labels_lane_line_color')   # 车道线的掩膜png所在的文件夹
    parser.add_argument('--sub_file', required=False, default='labels_drivable_area_sub') # 可驾驶区域减去车道线掩膜后的png所在的文件夹
    parser.add_argument('--sub_fusion_file', required=False, default='labels_drivable_area_sub_fusion')   # 可驾驶区域减去车道线掩膜后再叠加原图所在的文件夹
    args = parser.parse_args()
    image_file = args.image_file
    drivable_area_file = args.drivable_area_file
    lane_line_file = args.lane_line_file
    sub_file = args.sub_file
    sub_fusion_file = args.sub_fusion_file

    list = os.listdir(drivable_area_file)   # 获取image文件列表
    for i in range(0, len(list)):
        image_path = os.path.join(image_file, list[i])  # 获取每个jpg文件的绝对路径
        filename = list[i][:-4]       # 提取出.jpg前的字符作为文件名，以便后续使用
        print("NO", i)
        drivable_area_path = drivable_area_file+'/'+'{}.png'.format(filename)
        lane_line_path = lane_line_file+'/'+'{}.png'.format(filename)
        sub_path = sub_file+'/'+'{}.png'.format(filename)
        drivable_area_img = cv2.imread(drivable_area_path, cv2.IMREAD_COLOR) # 灰度图：IMREAD_GRAYSCALE
        lane_line_img = cv2.imread(lane_line_path, cv2.IMREAD_COLOR) # RBL：IMREAD_COLOR
        drivable_area_img[drivable_area_img>0] = 255
        lane_line_img[lane_line_img>0] = 255
        sub_img = drivable_area_img - lane_line_img
        sub_img[sub_img==255] = 128
        cv2.imwrite(sub_path,sub_img,[int(cv2.IMWRITE_PNG_COMPRESSION),9])
        print('Saved to: %s' % sub_file)

        # # 将mask掩膜和原图片重叠
        # fusion_path = sub_fusion_file+'/'+'{}.png'.format(filename)
        # # image = np.ascontiguousarray(image)
        # img1 = cv2.imread(image_path, cv2.IMREAD_COLOR)
        # img2 = cv2.imread(sub_path, cv2.IMREAD_COLOR)
        # combine = cv2.addWeighted(img1, 0.7,img2,0.3, 0)
        # cv2.imwrite(fusion_path,combine,[int(cv2.IMWRITE_PNG_COMPRESSION),9])
        # print('Saved to: %s' % sub_fusion_file)


if __name__ == '__main__':
    main()