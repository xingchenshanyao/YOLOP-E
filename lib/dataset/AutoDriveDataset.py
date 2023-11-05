import os
import cv2
import numpy as np
# np.set_printoptions(threshold=np.inf)
import random
import torch
import torchvision.transforms as transforms
# from visualization import plot_img_and_mask,plot_one_box,show_seg_result
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from ..utils import letterbox, augment_hsv, random_perspective, xyxy2xywh, cutout

# # ACE
# import os
# import math
# import matplotlib.pyplot as plt
# #线性拉伸处理
# #去掉最大最小0.5%的像素值 线性拉伸至[0,1]
# def stretchImage(data, s=0.005, bins = 2000):   
#     ht = np.histogram(data, bins)
#     d = np.cumsum(ht[0])/float(data.size)
#     lmin = 0; lmax=bins-1
#     while lmin<bins:
#         if d[lmin]>=s:
#             break
#         lmin+=1
#     while lmax>=0:
#         if d[lmax]<=1-s:
#             break
#         lmax-=1
#     return np.clip((data-ht[1][lmin])/(ht[1][lmax]-ht[1][lmin]), 0,1)
 
# #根据半径计算权重参数矩阵
# g_para = {}
# def getPara(radius = 5):                        
#     global g_para
#     m = g_para.get(radius, None)
#     if m is not None:
#         return m
#     size = radius*2+1
#     m = np.zeros((size, size))
#     for h in range(-radius, radius+1):
#         for w in range(-radius, radius+1):
#             if h==0 and w==0:
#                 continue
#             m[radius+h, radius+w] = 1.0/math.sqrt(h**2+w**2)
#     m /= m.sum()
#     g_para[radius] = m
#     return m
 
# #常规的ACE实现
# def zmIce(I, ratio=4, radius=300):                     
#     para = getPara(radius)
#     height,width = I.shape
#     zh = []
#     zw = []
#     n = 0
#     while n < radius:
#         zh.append(0)
#         zw.append(0)
#         n += 1
#     for n in range(height):
#         zh.append(n)
#     for n in range(width):
#         zw.append(n)
#     n = 0
#     while n < radius:
#         zh.append(height-1)
#         zw.append(width-1)
#         n += 1
#     #print(zh)
#     #print(zw)
    
#     Z = I[np.ix_(zh, zw)]
#     res = np.zeros(I.shape)
#     for h in range(radius*2+1):
#         for w in range(radius*2+1):
#             if para[h][w] == 0:
#                 continue
#             res += (para[h][w] * np.clip((I-Z[h:h+height, w:w+width])*ratio, -1, 1))
#     return res
 
# #单通道ACE快速增强实现
# def zmIceFast(I, ratio, radius):
#     # print(I)
#     height, width = I.shape[:2]
#     if min(height, width) <=2:
#         return np.zeros(I.shape)+0.5
#     Rs = cv2.resize(I, (int((width+1)/2), int((height+1)/2)))
#     Rf = zmIceFast(Rs, ratio, radius)             #递归调用
#     Rf = cv2.resize(Rf, (width, height))
#     Rs = cv2.resize(Rs, (width, height))
 
#     return Rf+zmIce(I,ratio, radius)-zmIce(Rs,ratio,radius)   
 
# #rgb三通道分别增强 ratio是对比度增强因子 radius是卷积模板半径          
# def zmIceColor(I, ratio=4, radius=3):               
#     res = np.zeros(I.shape)
#     for k in range(3):
#         res[:,:,k] = stretchImage(zmIceFast(I[:,:,k], ratio, radius))
#     return res




class AutoDriveDataset(Dataset):
    """
    A general Dataset for some common function
    """
    def __init__(self, cfg, is_train, inputsize=640, transform=None):
        """
        initial all the characteristic

        Inputs:
        -cfg: configurations
        -is_train(bool): whether train set or not
        -transform: ToTensor and Normalize
        
        Returns:
        None
        """
        self.is_train = is_train
        self.cfg = cfg
        self.transform = transform
        self.inputsize = inputsize
        self.Tensor = transforms.ToTensor()
        img_root = Path(cfg.DATASET.DATAROOT)
        label_root = Path(cfg.DATASET.LABELROOT)
        mask_root = Path(cfg.DATASET.MASKROOT)
        lane_root = Path(cfg.DATASET.LANEROOT)
        if is_train:
            indicator = cfg.DATASET.TRAIN_SET
        else:
            indicator = cfg.DATASET.TEST_SET
        self.img_root = img_root / indicator
        self.label_root = label_root / indicator
        self.mask_root = mask_root / indicator
        self.lane_root = lane_root / indicator
        # self.label_list = self.label_root.iterdir()
        self.mask_list = self.mask_root.iterdir()

        self.db = []

        self.data_format = cfg.DATASET.DATA_FORMAT

        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.flip = cfg.DATASET.FLIP
        self.color_rgb = cfg.DATASET.COLOR_RGB

        # self.target_type = cfg.MODEL.TARGET_TYPE
        self.shapes = np.array(cfg.DATASET.ORG_IMG_SIZE)
    
    def _get_db(self):
        """
        finished on children Dataset(for dataset which is not in Bdd100k format, rewrite children Dataset)
        """
        raise NotImplementedError

    def evaluate(self, cfg, preds, output_dir):
        """
        finished on children dataset
        """
        raise NotImplementedError
    
    def __len__(self,):
        """
        number of objects in the dataset
        """
        return len(self.db)

    def __getitem__(self, idx):
        """
        Get input and groud-truth from database & add data augmentation on input

        Inputs:
        -idx: the index of image in self.db(database)(list)
        self.db(list) [a,b,c,...]
        a: (dictionary){'image':, 'information':}

        Returns:
        -image: transformed image, first passed the data augmentation in __getitem__ function(type:numpy), then apply self.transform
        -target: ground truth(det_gt,seg_gt)

        function maybe useful
        cv2.imread
        cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        cv2.warpAffine
        """
        data = self.db[idx]
        data_label = data["label"]
        id_image = int(data["image"].split('/')[-1][:-4]) # 获取图片序号
        img = cv2.imread(data["image"], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        # cv2.imshow("img",img) # 原图像
        # cv2.waitKey(5000)
        
        # print("img = zmIceColor(img/255.0)*255")
        # img = zmIceColor(img/255.0)*255
        # cv2.imshow("img",img/255) # ACE自动色彩均衡快速算法
        # cv2.waitKey(5000)

        # Only Mascio Enhancement 数据增强 
        for line in data_label:
            idx_0 = line[0]
            x_c, y_c, w_c, h_c = int(line[1]*1280), int(line[2]*720), int(line[3]*1280), int(line[4]*720)
            x1, y1, x2, y2 = int(x_c-w_c/2), int(y_c-h_c/2), int(x_c+w_c/2), int(y_c+h_c/2)
            random.seed(idx)
            if self.is_train and int(idx_0) == 9 and random.random() > 1: # 只增强Straight or Right Turn Arrow
            # if self.is_train:
            # if True:
                c_y = 10 # 偏移间隙
                c_x = 0

                x_c_new = x_c+c_x
                y_c_new = y_c+h_c+c_y
                x1_new, y1_new, x2_new, y2_new = x1+c_x, y1+h_c+c_y, x2+c_x, y2+h_c+c_y

                if (x1_new >=0 and x2_new <=1280 and y1_new>=0 and  y2_new <=720):
                    # 向下重叠一次
                    Is_add = True
                    for line0 in data_label:
                        x1_0, y1_0, x2_0, y2_0 = line0[1]*1280-line0[3]*1280/2, line0[2]*1280-line0[4]*720/2, line0[1]*1280+line0[3]*1280/2, line0[2]*1280+line0[4]*720/2
                        if (x1_new>x1_0 and y1_new>y1_0 and x1_new<x2_0 and y1_new<y2_0) or (x2_new>x1_0 and y2_new>y1_0 and x2_new<x2_0 and y2_new<y2_0) or (x1_new<x1_0 and y1_new<y1_0 and x2_new>x2_0 and y2_new>y2_0):
                            Is_add = False
                            break
                    if Is_add:
                        try:
                            cropped_line = [[idx_0, x_c_new, y_c_new, w_c, h_c]]
                            data_label = np.append(data_label, cropped_line, axis=0)
                            img[int(y1_new):int(y2_new), int(x1_new):int(x2_new)] = img[int(y1):int(y2), int(x1):int(x2)]
                        except:
                            Is_add = True
                # cv2.imshow("img",img) 
                # cv2.waitKey(10000)

        # Specific Mascio Enhancement数据增强 
        cropped_path0 = '/home/xingchen/Study/dataset/SDExpressway/traffic_object_cropped/'
        f=open('/home/xingchen/Study/dataset/SDExpressway/traffic_object_cropped.txt','r')
        lines=f.readlines()
        f.close()
        c_c = 10
        p = 0.8 # 数据增强概率
        # Only_day = True
        Only_day = False #只加强白天的图片
        # if self.is_train: # 限定只有训练的时候增强
        # if True:
        if False:
            random.seed(idx)
            if random.random() > p-0.1 : # Straight or Right Turn Arrow增强
                Is_add = True
                if id_image >= 3294 and Only_day: # 只加强白天的图片
                    Is_add = False
                cropped_path = cropped_path0+'Straight or Right Turn Arrow/'
                fileList = os.listdir(cropped_path)
                cropped_id = random.randint(0,len(fileList)-1)
                txt_id = int(fileList[cropped_id].split('_')[0])
                txt_line = lines[txt_id-1].split(' ')
                x1, y1, x2, y2, idxx = int(txt_line[1]), int(txt_line[2]), int(txt_line[3]), int(txt_line[4]), int(txt_line[5])
                if x1>x2:
                    x1,x2 = x2,x1
                if y1>y2:
                    y1,y2 = y2,y1
                for line in data_label:
                    idx_0 = line[0]
                    x_c, y_c, w_c, h_c = int(line[1]*1280), int(line[2]*720), int(line[3]*1280), int(line[4]*720)
                    x1_0, y1_0, x2_0, y2_0 = int(x_c-w_c/2), int(y_c-h_c/2), int(x_c+w_c/2), int(y_c+h_c/2)
                    if (x1>x1_0 and y1>y1_0 and x1<x2_0 and y1<y2_0) or (x2>x1_0 and y2>y1_0 and x2<x2_0 and y2<y2_0) or (x1<x1_0 and y1<y1_0 and x2>x2_0 and y2>y2_0):
                        Is_add = False
                        break
                if Is_add:
                    try:
                        cropped = cv2.imread(cropped_path+fileList[cropped_id])
                        img[int(y1):int(y2), int(x1):int(x2)] = cropped
                        cropped_line = [[idxx, (x1+x2)/2/1280, (y1+y2)/2/720, (x2-x1)/1280, (y2-y1)/720]]
                        data_label = np.append(data_label, cropped_line, axis=0)
                    except:
                        Is_add = True

            if random.random() > p-0.1 : # Straight Ahead Arrow增强
                Is_add = True
                if id_image >= 3294 and Only_day: # 只加强白天的图片
                    Is_add = False
                cropped_path = cropped_path0+'Straight Ahead Arrow/'
                fileList = os.listdir(cropped_path)
                cropped_id = random.randint(0,len(fileList)-1)
                txt_id = int(fileList[cropped_id].split('_')[0])
                txt_line = lines[txt_id-1].split(' ')
                x1, y1, x2, y2, idxx = int(txt_line[1]), int(txt_line[2]), int(txt_line[3]), int(txt_line[4]), int(txt_line[5])
                if x1>x2:
                    x1,x2 = x2,x1
                if y1>y2:
                    y1,y2 = y2,y1
                for line in data_label:
                    idx_0 = line[0]
                    x_c, y_c, w_c, h_c = int(line[1]*1280), int(line[2]*720), int(line[3]*1280), int(line[4]*720)
                    x1_0, y1_0, x2_0, y2_0 = int(x_c-w_c/2), int(y_c-h_c/2), int(x_c+w_c/2), int(y_c+h_c/2)
                    if (x1>x1_0 and y1>y1_0 and x1<x2_0 and y1<y2_0) or (x2>x1_0 and y2>y1_0 and x2<x2_0 and y2<y2_0) or (x1<x1_0 and y1<y1_0 and x2>x2_0 and y2>y2_0):
                        Is_add = False
                        break
                if Is_add:
                    try:
                        cropped = cv2.imread(cropped_path+fileList[cropped_id])
                        img[int(y1):int(y2), int(x1):int(x2)] = cropped
                        cropped_line = [[idxx, (x1+x2)/2/1280, (y1+y2)/2/720, (x2-x1)/1280, (y2-y1)/720]]
                        data_label = np.append(data_label, cropped_line, axis=0)
                    except:
                        Is_add = True

            if random.random() > p : # Speed Limit Sign增强
                Is_add = True
                if id_image >= 3294 and Only_day: # 只加强白天的图片
                    Is_add = False
                cropped_path = cropped_path0+'Speed Limit Sign/'
                fileList = os.listdir(cropped_path)
                cropped_id = random.randint(0,len(fileList)-1)
                txt_id = int(fileList[cropped_id].split('_')[0])
                txt_line = lines[txt_id-1].split(' ')
                x1, y1, x2, y2, idxx = int(txt_line[1]), int(txt_line[2]), int(txt_line[3]), int(txt_line[4]), int(txt_line[5])
                if x1>x2:
                    x1,x2 = x2,x1
                if y1>y2:
                    y1,y2 = y2,y1
                for line in data_label:
                    idx_0 = line[0]
                    x_c, y_c, w_c, h_c = int(line[1]*1280), int(line[2]*720), int(line[3]*1280), int(line[4]*720)
                    x1_0, y1_0, x2_0, y2_0 = int(x_c-w_c/2), int(y_c-h_c/2), int(x_c+w_c/2), int(y_c+h_c/2)
                    if (x1>x1_0 and y1>y1_0 and x1<x2_0 and y1<y2_0) or (x2>x1_0 and y2>y1_0 and x2<x2_0 and y2<y2_0) or (x1<x1_0 and y1<y1_0 and x2>x2_0 and y2>y2_0):
                        Is_add = False
                        break
                if Is_add:
                    try:
                        cropped = cv2.imread(cropped_path+fileList[cropped_id])
                        img[max(0,int(y1-c_c)):min(720,int(y2+c_c)), max(0,int(x1-c_c)):min(1280,int(x2+c_c))] = cropped
                        cropped_line = [[idxx, (x1+x2)/2/1280, (y1+y2)/2/720, (x2-x1)/1280, (y2-y1)/720]]
                        data_label = np.append(data_label, cropped_line, axis=0)
                    except:
                        Is_add = True
                        

            if random.random() > p : # Emergency Telephone Sign增强
                Is_add = True
                if id_image >= 3294 and Only_day: # 只加强白天的图片
                    Is_add = False
                cropped_path = cropped_path0+'Emergency Telephone Sign/'
                fileList = os.listdir(cropped_path)
                cropped_id = random.randint(0,len(fileList)-1)
                txt_id = int(fileList[cropped_id].split('_')[0])
                txt_line = lines[txt_id-1].split(' ')
                x1, y1, x2, y2, idxx = int(txt_line[1]), int(txt_line[2]), int(txt_line[3]), int(txt_line[4]), int(txt_line[5])
                if x1>x2:
                    x1,x2 = x2,x1
                if y1>y2:
                    y1,y2 = y2,y1
                for line in data_label:
                    idx_0 = line[0]
                    x_c, y_c, w_c, h_c = int(line[1]*1280), int(line[2]*720), int(line[3]*1280), int(line[4]*720)
                    x1_0, y1_0, x2_0, y2_0 = int(x_c-w_c/2), int(y_c-h_c/2), int(x_c+w_c/2), int(y_c+h_c/2)
                    if (x1>x1_0 and y1>y1_0 and x1<x2_0 and y1<y2_0) or (x2>x1_0 and y2>y1_0 and x2<x2_0 and y2<y2_0) or (x1<x1_0 and y1<y1_0 and x2>x2_0 and y2>y2_0):
                        Is_add = False
                        break
                if Is_add:
                    try:
                        cropped = cv2.imread(cropped_path+fileList[cropped_id])
                        img[max(0,int(y1-c_c)):min(720,int(y2+c_c)), max(0,int(x1-c_c)):min(1280,int(x2+c_c))] = cropped
                        cropped_line = [[idxx, (x1+x2)/2/1280, (y1+y2)/2/720, (x2-x1)/1280, (y2-y1)/720]]
                        data_label = np.append(data_label, cropped_line, axis=0)
                    except:
                        Is_add = True

            if random.random() > p : # Warning Sign增强
                Is_add = True
                if id_image >= 3294 and Only_day: # 只加强白天的图片
                    Is_add = False
                cropped_path = cropped_path0+'Warning Sign/'
                fileList = os.listdir(cropped_path)
                cropped_id = random.randint(0,len(fileList)-1)
                txt_id = int(fileList[cropped_id].split('_')[0])
                txt_line = lines[txt_id-1].split(' ')
                x1, y1, x2, y2, idxx = int(txt_line[1]), int(txt_line[2]), int(txt_line[3]), int(txt_line[4]), int(txt_line[5])
                if x1>x2:
                    x1,x2 = x2,x1
                if y1>y2:
                    y1,y2 = y2,y1
                for line in data_label:
                    idx_0 = line[0]
                    x_c, y_c, w_c, h_c = int(line[1]*1280), int(line[2]*720), int(line[3]*1280), int(line[4]*720)
                    x1_0, y1_0, x2_0, y2_0 = int(x_c-w_c/2), int(y_c-h_c/2), int(x_c+w_c/2), int(y_c+h_c/2)
                    if (x1>x1_0 and y1>y1_0 and x1<x2_0 and y1<y2_0) or (x2>x1_0 and y2>y1_0 and x2<x2_0 and y2<y2_0) or (x1<x1_0 and y1<y1_0 and x2>x2_0 and y2>y2_0):
                        Is_add = False
                        break
                if Is_add:
                    try:
                        cropped = cv2.imread(cropped_path+fileList[cropped_id])
                        img[max(0,int(y1-c_c)):min(720,int(y2+c_c)), max(0,int(x1-c_c)):min(1280,int(x2+c_c))] = cropped
                        cropped_line = [[idxx, (x1+x2)/2/1280, (y1+y2)/2/720, (x2-x1)/1280, (y2-y1)/720]]
                        data_label = np.append(data_label, cropped_line, axis=0)
                    except:
                        Is_add = True

            if random.random() > p : # Directional Sign增强
                Is_add = True
                if id_image >= 3294 and Only_day: # 只加强白天的图片
                    Is_add = False
                cropped_path = cropped_path0+'Directional Sign/'
                fileList = os.listdir(cropped_path)
                cropped_id = random.randint(0,len(fileList)-1)
                txt_id = int(fileList[cropped_id].split('_')[0])
                txt_line = lines[txt_id-1].split(' ')
                x1, y1, x2, y2, idxx = int(txt_line[1]), int(txt_line[2]), int(txt_line[3]), int(txt_line[4]), int(txt_line[5])
                if x1>x2:
                    x1,x2 = x2,x1
                if y1>y2:
                    y1,y2 = y2,y1
                for line in data_label:
                    idx_0 = line[0]
                    x_c, y_c, w_c, h_c = int(line[1]*1280), int(line[2]*720), int(line[3]*1280), int(line[4]*720)
                    x1_0, y1_0, x2_0, y2_0 = int(x_c-w_c/2), int(y_c-h_c/2), int(x_c+w_c/2), int(y_c+h_c/2)
                    if (x1>x1_0 and y1>y1_0 and x1<x2_0 and y1<y2_0) or (x2>x1_0 and y2>y1_0 and x2<x2_0 and y2<y2_0) or (x1<x1_0 and y1<y1_0 and x2>x2_0 and y2>y2_0):
                        Is_add = False
                        break
                if Is_add:
                    try:
                        cropped = cv2.imread(cropped_path+fileList[cropped_id])
                        img[max(0,int(y1-c_c)):min(720,int(y2+c_c)), max(0,int(x1-c_c)):min(1280,int(x2+c_c))] = cropped
                        cropped_line = [[idxx, (x1+x2)/2/1280, (y1+y2)/2/720, (x2-x1)/1280, (y2-y1)/720]]
                        data_label = np.append(data_label, cropped_line, axis=0)
                    except:
                        Is_add = True

            if random.random() > p : # Pending Sign增强
                Is_add = True
                if id_image >= 3294 and Only_day: # 只加强白天的图片
                    Is_add = False
                cropped_path = cropped_path0+'Pending Sign/'
                fileList = os.listdir(cropped_path)
                cropped_id = random.randint(0,len(fileList)-1)
                txt_id = int(fileList[cropped_id].split('_')[0])
                txt_line = lines[txt_id-1].split(' ')
                x1, y1, x2, y2, idxx = int(txt_line[1]), int(txt_line[2]), int(txt_line[3]), int(txt_line[4]), int(txt_line[5])
                if x1>x2:
                    x1,x2 = x2,x1
                if y1>y2:
                    y1,y2 = y2,y1
                for line in data_label:
                    idx_0 = line[0]
                    x_c, y_c, w_c, h_c = int(line[1]*1280), int(line[2]*720), int(line[3]*1280), int(line[4]*720)
                    x1_0, y1_0, x2_0, y2_0 = int(x_c-w_c/2), int(y_c-h_c/2), int(x_c+w_c/2), int(y_c+h_c/2)
                    if (x1>x1_0 and y1>y1_0 and x1<x2_0 and y1<y2_0) or (x2>x1_0 and y2>y1_0 and x2<x2_0 and y2<y2_0) or (x1<x1_0 and y1<y1_0 and x2>x2_0 and y2>y2_0):
                        Is_add = False
                        break
                if Is_add:
                    try:
                        cropped = cv2.imread(cropped_path+fileList[cropped_id])
                        img[max(0,int(y1-c_c)):min(720,int(y2+c_c)), max(0,int(x1-c_c)):min(1280,int(x2+c_c))] = cropped
                        cropped_line = [[idxx, (x1+x2)/2/1280, (y1+y2)/2/720, (x2-x1)/1280, (y2-y1)/720]]
                        data_label = np.append(data_label, cropped_line, axis=0)
                    except:
                        Is_add = True

            if random.random() > p : # Guidance Sign增强
                Is_add = True
                if id_image >= 3294 and Only_day: # 只加强白天的图片
                    Is_add = False
                cropped_path = cropped_path0+'Guidance Sign/'
                fileList = os.listdir(cropped_path)
                cropped_id = random.randint(0,len(fileList)-1)
                txt_id = int(fileList[cropped_id].split('_')[0])
                txt_line = lines[txt_id-1].split(' ')
                x1, y1, x2, y2, idxx = int(txt_line[1]), int(txt_line[2]), int(txt_line[3]), int(txt_line[4]), int(txt_line[5])
                if x1>x2:
                    x1,x2 = x2,x1
                if y1>y2:
                    y1,y2 = y2,y1
                for line in data_label:
                    idx_0 = line[0]
                    x_c, y_c, w_c, h_c = int(line[1]*1280), int(line[2]*720), int(line[3]*1280), int(line[4]*720)
                    x1_0, y1_0, x2_0, y2_0 = int(x_c-w_c/2), int(y_c-h_c/2), int(x_c+w_c/2), int(y_c+h_c/2)
                    if (x1>x1_0 and y1>y1_0 and x1<x2_0 and y1<y2_0) or (x2>x1_0 and y2>y1_0 and x2<x2_0 and y2<y2_0) or (x1<x1_0 and y1<y1_0 and x2>x2_0 and y2>y2_0):
                        Is_add = False
                        break
                if Is_add:
                    try:
                        cropped = cv2.imread(cropped_path+fileList[cropped_id])
                        img[max(0,int(y1-c_c)):min(720,int(y2+c_c)), max(0,int(x1-c_c)):min(1280,int(x2+c_c))] = cropped
                        cropped_line = [[idxx, (x1+x2)/2/1280, (y1+y2)/2/720, (x2-x1)/1280, (y2-y1)/720]]
                        data_label = np.append(data_label, cropped_line, axis=0)
                    except:
                        Is_add = True


        data["label"] = data_label
        # cv2.imshow("img",img) 
        # cv2.waitKey(10000)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # cv2.imshow("img",img) # 图像颜色空间转换
        # cv2.waitKey(10000)

        # seg_label = cv2.imread(data["mask"], 0)
        if self.cfg.num_seg_class == 3:
            seg_label = cv2.imread(data["mask"])
        else:
            seg_label = cv2.imread(data["mask"], 0)
        lane_label = cv2.imread(data["lane"], 0)
        #print(lane_label.shape)
        # print(seg_label.shape)
        # print(lane_label.shape)
        # print(seg_label.shape)
        resized_shape = self.inputsize
        if isinstance(resized_shape, list):
            resized_shape = max(resized_shape)
        h0, w0 = img.shape[:2]  # orig hw
        r = resized_shape / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
            # cv2.imshow("img",img) # 图像缩小到640*360
            # cv2.waitKey(10000)
            seg_label = cv2.resize(seg_label, (int(w0 * r), int(h0 * r)), interpolation=interp)
            lane_label = cv2.resize(lane_label, (int(w0 * r), int(h0 * r)), interpolation=interp)
        h, w = img.shape[:2]
        
        (img, seg_label, lane_label), ratio, pad = letterbox((img, seg_label, lane_label), resized_shape, auto=True, scaleup=self.is_train)
        shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling
        # ratio = (w / w0, h / h0)
        # print(resized_shape)
        
        det_label = data["label"]
        labels=[]
        
        if det_label.size > 0:
            # Normalized xywh to pixel xyxy format
            labels = det_label.copy()
            labels[:, 1] = ratio[0] * w * (det_label[:, 1] - det_label[:, 3] / 2) + pad[0]  # pad width
            labels[:, 2] = ratio[1] * h * (det_label[:, 2] - det_label[:, 4] / 2) + pad[1]  # pad height
            labels[:, 3] = ratio[0] * w * (det_label[:, 1] + det_label[:, 3] / 2) + pad[0]
            labels[:, 4] = ratio[1] * h * (det_label[:, 2] + det_label[:, 4] / 2) + pad[1]
            
        if self.is_train:
            combination = (img, seg_label, lane_label)
            (img, seg_label, lane_label), labels = random_perspective(
                combination=combination,
                targets=labels,
                degrees=self.cfg.DATASET.ROT_FACTOR,
                translate=self.cfg.DATASET.TRANSLATE,
                scale=self.cfg.DATASET.SCALE_FACTOR,
                shear=self.cfg.DATASET.SHEAR
            )
            #print(labels.shape)
            augment_hsv(img, hgain=self.cfg.DATASET.HSV_H, sgain=self.cfg.DATASET.HSV_S, vgain=self.cfg.DATASET.HSV_V)
            # img, seg_label, labels = cutout(combination=combination, labels=labels)

            if len(labels):
                # convert xyxy to xywh
                labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])

                # Normalize coordinates 0 - 1
                labels[:, [2, 4]] /= img.shape[0]  # height
                labels[:, [1, 3]] /= img.shape[1]  # width

            # if self.is_train:
            # random left-right flip
            lr_flip = True
            if lr_flip and random.random() < 0.5:
                img = np.fliplr(img)
                seg_label = np.fliplr(seg_label)
                lane_label = np.fliplr(lane_label)
                if len(labels):
                    labels[:, 1] = 1 - labels[:, 1]

            # random up-down flip
            ud_flip = False
            if ud_flip and random.random() < 0.5:
                img = np.flipud(img)
                seg_label = np.filpud(seg_label)
                lane_label = np.filpud(lane_label)
                if len(labels):
                    labels[:, 2] = 1 - labels[:, 2]
        
        else:
            if len(labels):
                # convert xyxy to xywh
                labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])

                # Normalize coordinates 0 - 1
                labels[:, [2, 4]] /= img.shape[0]  # height
                labels[:, [1, 3]] /= img.shape[1]  # width

        labels_out = torch.zeros((len(labels), 6))
        if len(labels):
            labels_out[:, 1:] = torch.from_numpy(labels)
        # Convert
        # img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        # img = img.transpose(2, 0, 1)
        # cv2.imshow("img", img)
        # cv2.waitKey(10000)
        img = np.ascontiguousarray(img) # 返回一个连续的array

        # seg_label = np.ascontiguousarray(seg_label)
        # if idx == 0:
        #     print(seg_label[:,:,0])

        if self.cfg.num_seg_class == 3:
            _,seg0 = cv2.threshold(seg_label[:,:,0],128,255,cv2.THRESH_BINARY)
            _,seg1 = cv2.threshold(seg_label[:,:,1],1,255,cv2.THRESH_BINARY)
            _,seg2 = cv2.threshold(seg_label[:,:,2],1,255,cv2.THRESH_BINARY)
        else:
            _,seg1 = cv2.threshold(seg_label,1,255,cv2.THRESH_BINARY)
            _,seg2 = cv2.threshold(seg_label,1,255,cv2.THRESH_BINARY_INV)
        _,lane1 = cv2.threshold(lane_label,1,255,cv2.THRESH_BINARY)
        _,lane2 = cv2.threshold(lane_label,1,255,cv2.THRESH_BINARY_INV)
#        _,seg2 = cv2.threshold(seg_label[:,:,2],1,255,cv2.THRESH_BINARY)
        # # seg1[cutout_mask] = 0
        # # seg2[cutout_mask] = 0
        
        # seg_label /= 255
        # seg0 = self.Tensor(seg0)
        if self.cfg.num_seg_class == 3:
            seg0 = self.Tensor(seg0)
        seg1 = self.Tensor(seg1)
        seg2 = self.Tensor(seg2)
        # seg1 = self.Tensor(seg1)
        # seg2 = self.Tensor(seg2)
        lane1 = self.Tensor(lane1)
        lane2 = self.Tensor(lane2)

        # seg_label = torch.stack((seg2[0], seg1[0]),0)
        if self.cfg.num_seg_class == 3:
            seg_label = torch.stack((seg0[0],seg1[0],seg2[0]),0)
        else:
            seg_label = torch.stack((seg2[0], seg1[0]),0)
            
        lane_label = torch.stack((lane2[0], lane1[0]),0)
        # _, gt_mask = torch.max(seg_label, 0)
        # _ = show_seg_result(img, gt_mask, idx, 0, save_dir='debug', is_gt=True)
        

        target = [labels_out, seg_label, lane_label]
        # cv2.imshow("img", img) # 这里img还是图像
        # cv2.waitKey(10000)
        # print(img)
        img = self.transform(img) # 这里img变成了数组的格式
        # print(img)

        return img, target, data["image"], shapes

    def select_data(self, db):
        """
        You can use this function to filter useless images in the dataset

        Inputs:
        -db: (list)database

        Returns:
        -db_selected: (list)filtered dataset
        """
        db_selected = ...
        return db_selected

    @staticmethod
    def collate_fn(batch):
        img, label, paths, shapes= zip(*batch)
        label_det, label_seg, label_lane = [], [], []
        for i, l in enumerate(label):
            l_det, l_seg, l_lane = l
            l_det[:, 0] = i  # add target image index for build_targets()
            label_det.append(l_det)
            label_seg.append(l_seg)
            label_lane.append(l_lane)
        return torch.stack(img, 0), [torch.cat(label_det, 0), torch.stack(label_seg, 0), torch.stack(label_lane, 0)], paths, shapes

