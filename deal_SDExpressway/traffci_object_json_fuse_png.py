import os
import json
import colorsys
import numpy as np
import torch.nn as nn
from PIL import ImageDraw, ImageFont,Image

class Add_box:
    def __init__(self):
        
        self.bdd100k_width_ratio = 1.0/1280
        self.bdd100k_height_ratio = 1.0/720
        self.select_categorys = ["Car", "Truck", "Guidance Sign", "Warning Sign", "Pending Sign", "Speed Limit Sign", "Emergency Telephone Sign", "Directional Sign", "Straight Ahead Arrow", "Straight or Right Turn Arrow","Nothing"]
        self.traffic_signs = ["Guidance Sign", "Warning Sign", "Pending Sign", "Speed Limit Sign", "Emergency Telephone Sign", "Directional Sign"]
        self.ground_signs = ["Straight Ahead Arrow", "Straight or Right Turn Arrow"]
        self.num_classes = 11
        self.categorys_nums = {
            'Car':0,
            'Truck':1,
            'Guidance Sign':2,
            'Warning Sign':3,
            'Pending Sign':4,
            'Speed Limit Sign':5,
            'Emergency Telephone Sign':6,
            'Directional Sign':7,
            'Straight Ahead Arrow':8,
            'Straight or Right Turn Arrow':9,
            'Nothing':10
        }
        self.keys = self.categorys_nums.keys()
        #   画框设置不同的颜色
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))


    def detect_image(self,images_path, lines, num):
            #   计算输入图片的高和宽
            image = Image.open(images_path)
            print(lines)

            #   设置字体与边框厚度
            font        = ImageFont.truetype(font='simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
            thickness   = int(2)

            #   图像绘制
            for i in list(enumerate(lines)):
                predicted_class = list(self.keys)[i[1][4]]
                c = int(i[1][4])

                #i[1][0]=x1=left
                #i[1][1]=y1=top
                #i[1][2]=x2=right 
                #i[1][3]=y2=bottom
                top = i[1][1] #top是y1 left是x1 right是x2 bottom是y2
                left = i[1][0]
                bottom = i[1][3]
                right = i[1][2]

                top     = max(0, np.floor(top).astype('int32'))
                left    = max(0, np.floor(left).astype('int32'))
                bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
                right   = min(image.size[0], np.floor(right).astype('int32'))

                label = '{} '.format(predicted_class)
                draw = ImageDraw.Draw(image)
                label_size = draw.textsize(label, font)
                label = label.encode('utf-8')
                # print(label, top, left, bottom, right)
                
                if top - label_size[1] >= 0:
                    text_origin = np.array([left, top - label_size[1]])
                else:
                    text_origin = np.array([left, top + 1])

                for j in range(thickness):
                    draw.rectangle([left + j, top + j, right - j, bottom - j], outline=self.colors[c])
                # draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
                # draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
                del draw
            
            return image

    def add_box(self, labels_path, images_path, dir_save_path ):
        lines = [] #lines是一个二维列表，用于存放一张图片所有的真值框+类别
        num = 0
        with open(labels_path) as fp:
            j = json.load(fp)
            for fr in j["shapes"]:
                temp_category=fr["label"]
                idx = self.categorys_nums[temp_category]

                if fr["label"] in self.select_categorys:
                    # cx = (objs["box2d"]["x1"] + objs["box2d"]["x2"]) / 2.0
                    # cy = (objs["box2d"]["y1"] + objs["box2d"]["y2"]) / 2.0
                    # w = objs["box2d"]["x2"] - objs["box2d"]["x1"]
                    # h = objs["box2d"]["y2"] - objs["box2d"]["y1"]

                    #(x1,y1)为左上角点，(x2,y2)为右下角点
                    x1=fr["points"][0][0] 
                    x2=fr["points"][1][0] 
                    y1=fr["points"][0][1]
                    y2=fr["points"][1][1]

                    # 确保右下角点坐标大于左上角点
                    if x1>x2:
                        x1, x2 = x2, x1 
                    if y1>y2:
                        y1, y2 = y2, y1

                    line = [] #line是一个一维列表，用于存放一个真值框+类别
                    line.append(x1)
                    line.append(y1)
                    line.append(x2)
                    line.append(y2)
                    line.append(idx)
                    lines.append(line)
                    num = num +1


                
                r_image  = self.detect_image(images_path, lines, num)
                r_image.save(os.path.join(dir_save_path, j["imagePath"][:-4]+".png"), quality=95, subsampling=0)
                if num != 0:
                    print("%s has been drawn!" % j["imagePath"][:-4])
                

if __name__ == "__main__":
    #加载labels与images文件夹
    labels_dir = "labels_traffic_object_json" #标签文件夹
    images_dir = "images" #图片文件夹
    dir_save_path = "labels_traffic_object_fusion" #输出图片保存文件夹

    #剔除fileList中的非json文件
    fileList = os.listdir(labels_dir)
    jsons= []
    names=[]
    for json1 in fileList:
        if json1.endswith(".json"):
            jsons.append(json1)

    #names中是去掉".json"的文件名的集合
    num = len(jsons)
    list0    = range(num)
    for i in list0:  
        name=jsons[i][:-5] 
        names.append(name)

    obj = Add_box()
    for path in names:
        labels_path = labels_dir+'/'+path+'.json'
        images_path = images_dir +'/'+path+'.jpg' #注意这里图片的格式是jpg
        print(path)
        obj.add_box(labels_path, images_path, dir_save_path )