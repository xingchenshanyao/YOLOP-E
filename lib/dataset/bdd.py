import numpy as np
import json

from .AutoDriveDataset import AutoDriveDataset
from .convert import convert, id_dict, id_dict_single, id_dict_SDExpressway, id_dict_SDExpressway_single
from tqdm import tqdm

single_cls = False       # just detect vehicle 

class BddDataset(AutoDriveDataset):
    def __init__(self, cfg, is_train, inputsize, transform=None):
        super().__init__(cfg, is_train, inputsize, transform) 
        self.db = self._get_db() # 加载数据集 # self.db = [{'image': '/home/xingchen/Study...3225df.jpg', 'label': array([[0.        , ...7547441]]), 'mask': '/home/xingchen/Study...3225df.png', 'lane': '/home/xingchen/Study...3225df.png'},  ...]
        self.cfg = cfg

    def _get_db(self):
        """
        get database from the annotation file

        Inputs:

        Returns:
        gt_db: (list)database   [a,b,c,...]
                a: (dictionary){'image':, 'information':, ......}
        image: image path
        mask: path of the segmetation label
        label: [cls_id, center_x//256, center_y//256, w//256, h//256] 256=IMAGE_SIZE
        """
        print('building database...')
        gt_db = []
        height, width = self.shapes
        for mask in tqdm(list(self.mask_list)): # 加载数据集和标签
            mask_path = str(mask)
            label_path = mask_path.replace(str(self.mask_root), str(self.label_root)).replace(".png", ".json")
            image_path = mask_path.replace(str(self.mask_root), str(self.img_root)).replace(".png", ".jpg")
            lane_path = mask_path.replace(str(self.mask_root), str(self.lane_root))
            with open(label_path, 'r') as f:
                label = json.load(f)

            # # BDD100k
            # data = label['frames'][0]['objects']
            # data = self.filter_data(data)
            # gt = np.zeros((len(data), 5))
            # for idx, obj in enumerate(data):
            #     category = obj['category']
            #     if category == "traffic light":
            #         color = obj['attributes']['trafficLightColor']
            #         category = "tl_" + color
            #     if category in id_dict.keys():
            #         x1 = float(obj['box2d']['x1'])
            #         y1 = float(obj['box2d']['y1'])
            #         x2 = float(obj['box2d']['x2'])
            #         y2 = float(obj['box2d']['y2'])
            #         cls_id = id_dict[category]
            #         gt[idx][0] = cls_id
            #         box = convert((width, height), (x1, x2, y1, y2))
            #         gt[idx][1:] = list(box)

            # SDExpressway
            data = label['shapes'] 
            data = self.filter_data(data)
            gt = np.zeros((len(data), 5))
            for idx, obj in enumerate(data):
                category = obj['label'] # 类别
                if category in id_dict_SDExpressway.keys():
                    x1 = float(obj['points'][0][0])
                    y1 = float(obj['points'][0][1])
                    x2 = float(obj['points'][1][0])
                    y2 = float(obj['points'][1][1])
                    if x1>x2:
                        x1, x2 = x2, x1
                    if y1>y2:
                        y1, y2 = y2, y1
                    cls_id = id_dict_SDExpressway[category]
                    # if single_cls: # 20230816
                    #      cls_id=0
                    gt[idx][0] = cls_id
                    box = convert((width, height), (x1, x2, y1, y2))
                    gt[idx][1:] = list(box)
                

            rec = [{
                'image': image_path,
                'label': gt,
                'mask': mask_path,
                'lane': lane_path
            }]

            gt_db += rec
        print('database build finish')
        return gt_db

    # # BDD100k数据集
    # def filter_data(self, data):
    #     remain = []
    #     for obj in data:
    #         if 'box2d' in obj.keys():  # obj.has_key('box2d'):
    #             if single_cls: # 只预测车辆
    #                 if obj['category'] in id_dict_single.keys():
    #                     remain.append(obj)
    #             else:
    #                 remain.append(obj)
    #     return remain

    # SDExpressway数据集
    def filter_data(self, data):
        remain = []
        for obj in data:
            if 'points' in obj.keys():  # obj.has_key('box2d'):
                if single_cls: 
                    if obj['label'] in id_dict_SDExpressway_single.keys():
                        remain.append(obj)
                else:
                    remain.append(obj)
        return remain

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        """  
        """
        pass
