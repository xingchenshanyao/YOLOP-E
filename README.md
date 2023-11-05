## YOLOP-E: You Only Look Once for Expressway Panoramic Driving Perception

Our paper: [YOLOP-E: You Only Look Once for Expressway Panoramic Driving Perception](https://github.com/hustvl/YOLOP)

Our datasets and weights: [SDExpressway, yolop.pth, yolope.pth](https://pan.baidu.com/s/1589VGpmHATSrTs6f_HSI_g?pwd=m2jh)

### The Illustration of YOLOP-E
![图片1](https://github.com/xingchenshanyao/YOLOP-E/assets/116085226/5588ffc2-4b3e-40e2-92a5-429ec596c03b)

### Contributions
* 基于YOLOP提出了一种高效的针对高速公路场景的多任务网络YOLOP-E，可联合处理(jointly handle)自动驾驶中的三项关键任务：交通目标检测、可驾驶区域分割和车道线分割，并且达到优异的检测与分割效果。

* 采集并制作了SDExpressway数据集，其包含晴天、黑夜、雨雾天气下的5603张图片。我们在SDExpressway与BDD100k上对所提网络进行了广泛评估，进行了消融实验与SOTA对比实验，以证明所提模型各项改进的有效性以及对高速公路场景检测的适应性，证实它即使在恶劣环境下也具有良好的鲁棒性和泛化能力。

* 优化了ELAN模块得到更高效的聚合网络结构ELAN-X，结合了更多不同深度的特征信息进行并行处理，提升模型的感受野和特征表达能力，提高了多任务模型检测的精度。

### Results

#### On the SDExpressway

##### Traffic Object Detection Result
![1](https://github.com/xingchenshanyao/YOLOP-E/assets/116085226/531610d1-e956-4b5a-bdda-60a822a4eda1)
![8](https://github.com/xingchenshanyao/YOLOP-E/assets/116085226/1d62660f-9e20-4754-98ff-6392409e5e36)
##### Drivable Area Segmentation Result
![3](https://github.com/xingchenshanyao/YOLOP-E/assets/116085226/f04437b1-b439-44e1-90e5-e983e448174f)
##### Lane Detection Result
![5](https://github.com/xingchenshanyao/YOLOP-E/assets/116085226/b0ed4927-69d4-42eb-8bae-6db30f30019e)

##### On the BDD100k

##### Traffic Object Detection Result
![2](https://github.com/xingchenshanyao/YOLOP-E/assets/116085226/bda50edb-67bc-4fb1-a3f5-403920cd3a6c)
##### Drivable Area Segmentation Result
![4](https://github.com/xingchenshanyao/YOLOP-E/assets/116085226/8b608795-779d-4b6f-b71a-0a26789776b7)
##### Lane Detection Result:
![6](https://github.com/xingchenshanyao/YOLOP-E/assets/116085226/622348b1-5f9a-4eb5-8bf1-f0d3fb5f9548)

### Visualization

#### Traffic Object Detection Result
![1](https://github.com/xingchenshanyao/YOLOP-E/assets/116085226/eb4e0e3a-296f-4484-aedc-07c362cbd6fb)
#### Drivable Area Segmentation Result
![2](https://github.com/xingchenshanyao/YOLOP-E/assets/116085226/33071530-7695-4cb5-b91a-77411c032dcf)
#### Lane Detection Result
![3](https://github.com/xingchenshanyao/YOLOP-E/assets/116085226/7a4909a2-b70b-4161-a4d9-0b8a62ef7d4f)


### Acknowledgements:

[YOLOP](https://github.com/hustvl/YOLOP)
