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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_file', required=False, default='images')   # 原图image所在的文件夹
    parser.add_argument('--json_file', required=False, default='labels_drivable_area_json')   # 标注文件json所在的文件夹
    parser.add_argument('--black_file', required=False, default='labels_drivable_area_black')   # 不可见的掩膜png所在的文件夹
    parser.add_argument('--color_file', required=False, default='labels_drivable_area_color')   # 可见的掩膜png所在的文件夹
    parser.add_argument('--fusion_file', required=False, default='labels_drivable_area_fusion')   # 掩膜与原图叠加后的png所在的文件夹
    args = parser.parse_args()
    image_file = args.image_file
    json_file = args.json_file
    black_file = args.black_file
    color_file = args.color_file
    fusion_file = args.fusion_file

    list = os.listdir(image_file)   # 获取image文件列表
    for i in range(0, len(list)):
        image_path = os.path.join(image_file, list[i])  # 获取每个jpg文件的绝对路径
        filename = list[i][:-4]       # 提取出.jpg前的字符作为文件名，以便后续使用
        json_path = os.path.join(json_file, filename+'.json') # 获取每个json文件的绝对路径
        print("NO", i)
        extension = list[i][-3:]
        if extension == 'jpg':
            if os.path.isfile(json_path):

                # 生成不可见的mask掩膜图片
                data = json.load(open(json_path))
                img = utils.image.img_b64_to_arr(data['imageData'])  # 根据'imageData'字段的字符可以得到原图像
                # lbl为label图片（标注的地方用类别名对应的数字来标，其他为0）lbl_names为label名和数字的对应关系字典
                lbl, lbl_names = utils.shape.labelme_shapes_to_label(img.shape, data['shapes'])   # data['shapes']是json文件中记录着标注的位置及label等信息的字段
                PIL.Image.fromarray(lbl).save(osp.join(black_file, '{}.png'.format(filename)))
                print('Saved to: %s' % black_file)

                # 将mask掩膜可视化
                black_path = os.path.join(black_file, '{}.png'.format(filename))
                mask = Image.open(black_path).convert('L')
                mask.putpalette([0, 0, 0, 0, 128, 0]) #　putpalette给对象加上调色板，相当于上色
                color_path = color_file+'/'+'{}.png'.format(filename)
                mask.save(color_path)
                print('Saved to: %s' % color_file)

                # # 将mask掩膜和原图片重叠
                # fusion_path = fusion_file+'/'+'{}.png'.format(filename)
                # # image = np.ascontiguousarray(image)
                # img1 = cv2.imread(image_path, cv2.IMREAD_COLOR)
                # img2 = cv2.imread(color_path, cv2.IMREAD_COLOR)
                # combine = cv2.addWeighted(img1, 1,img2,0.3, 0)
                # cv2.imwrite(fusion_path,combine,[int(cv2.IMWRITE_PNG_COMPRESSION),9])
                # print('Saved to: %s' % fusion_file)


if __name__ == '__main__':
    main()