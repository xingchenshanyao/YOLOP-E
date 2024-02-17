## YOLOP-E: You Only Look Once for Expressway Panoramic Driving Perception

Our paper: YOLOP-E: You Only Look Once for Expressway Panoramic Driving Perception(Submitted)

Our datasets and weights: [SDExpressway, yolop.pth, yolope.pth](https://pan.baidu.com/s/1589VGpmHATSrTs6f_HSI_g)

***

### The Illustration of YOLOP-E
![1](https://github.com/xingchenshanyao/YOLOP-E/assets/116085226/8e115e1f-a5e3-4861-a58b-7f17de362344)

### The Illustration of ELAN-X
![ELAN-X](https://github.com/xingchenshanyao/YOLOP-E/assets/116085226/98e7bdb3-936a-4994-ab2e-75078b2ea0be)

***

### Contributions
* This study has produced the expressway multi-task dataset, **SDExpressway**, encompassing 5603 images captured under various weather conditions, including sunny, dark, rainy and foggy scenarios. Each image in the dataset has been meticulously labeled with drivable areas, lane lines, and traffic object information.

* This research endeavors include the optimization of the ELAN module, resulting in the creation of a more efficient aggregated network structure known as **ELAN-X**. This innovation facilitates the amalgamation of feature information from various depths for parallel processing, enhancing the sensory field and feature expression of the model. These enhancements bolster the accuracy of the multi-task model's detection capabilities.

* This paper introduce an efficient multi-task network, **YOLOP-E**, tailored for expressway scenarios and built upon the YOLOP framework. YOLOP-E is engineered to jointly handle three critical tasks in autonomous driving: traffic object detection, drivable area segmentation, and lane line segmentation.

*  The proposed network undergoes extensive evaluation on both the SDExpressway dataset and the widely recognized BDD100k dataset, including ablation experiments and state-of-the-art (SOTA) comparison experiments to demonstrate the efficacy of the various improvements integrated into the model. Notably, the proposed model showcases robustness and strong generalization abilities, even in challenging environmental conditions.

***

### Results

#### On the SDExpressway

##### Traffic Object Detection Result
| Network          | R(%) | mAP50(%) | mAP50:95(%) | FPS(fps) |
| :--------------: | :---------: | :--------: | :----------: | :----------: |
| `YOLOP(baseline)`     | 86.8      | 74.4     | 38.7     | 232        |
| `HybridNets`      | 90.1      | 76.4     | 42.1     | 110        |
| `YOLOP-E(ours)`  | 92.1(**+5.3**)      | 83.8(**+9.4**)     | 53.3(**+14.6**)     | 127         |

![3](https://github.com/xingchenshanyao/YOLOP-E/assets/116085226/31bbe3c1-ca55-444a-b735-afabdfc02be3)

##### Drivable Area Segmentation Result
| Network          | mIoU(%) | FPS(fps) |
| :--------------: | :---------: | :--------: |
| `YOLOP(baseline)`     | 97.7      | 232     |
| `HybridNets`      | 97.5      | 110     |
| `YOLOP-E(ours)`  | 98.1(**+0.4**)      | 127     |
##### Lane Detection Result
| Network          | Acc(%) |IoU(%) | FPS(fps) |
| :--------------: | :---------: | :--------: | :--------: |
| `YOLOP(baseline)`     | 90.8      | 72.8     | 232     |
| `HybridNets`      | 92.0      | 75.7     | 110    |
| `YOLOP-E(ours)`  | 92.1(**+1.3**)      | 76.2(**+3.4**)     | 127     |

#### On the BDD100k

##### Traffic Object Detection Result
| Network          | R(%) | mAP50(%) | mAP50:95(%) | FPS(fps) |
| :--------------: | :---------: | :--------: | :----------: | :----------: |
| `YOLOP(baseline)`     | 89.5      | 76.3     | 43.1     | 230        |
| `MultiNet`      | 81.3      | 60.2     | 33.1     | 51        |
| `DLT-Net`  | 89.4      | 68.4     | 38.6     | 56         |
| `HybridNets`      | 92.8      | 77.3     | 45.8     | 108        |
| `YOLOP-E(ours)`  | 92.0(**+2.5**)      | 79.7(**+3.4**)     | 46.7(**+3.6**)     | 120         |
##### Drivable Area Segmentation Result
| Network          | mIoU(%) | FPS(fps) |
| :--------------: | :---------: | :--------: |
| `YOLOP(baseline)`     | 91.3      | 230     |
| `MultiNet`      | 71.6     | 51     |
| `DLT-Net`  | 71.3     | 56     |
| `HybridNets`      | 90.5      | 108     |
| `YOLOP-E(ours)`  | 92.1(**+0.8**)      | 120     |
##### Lane Detection Result
| Network          | Acc(%) |IoU(%) | FPS(fps) |
| :--------------: | :---------: | :--------: | :--------: |
| `YOLOP(baseline)`     | 70.5      | 26.2    | 230     |
| `HybridNets`      | 85.4      | 31.6     | 108   |
| `YOLOP-E(ours)`  | 73.0 (**+2.5**)     | 27.3(**+1.1**)     | 120     |

#### The evaluation of effificient experiments
![1](https://github.com/xingchenshanyao/YOLOP-E/assets/116085226/03b85a7b-720d-4820-a2e5-3ee8c6eca162)

#### The comparison of performance effects of adding SimAM attention mechanisms at different locations
![2](https://github.com/xingchenshanyao/YOLOP-E/assets/116085226/ea253137-2d2e-4e61-bcf6-5247635c71ad)

***

### Visualization

NOTE：YOLOP (left), HybridNets (center), and YOLOP-E (right) 

#### Traffic Object Detection Result
![1](https://github.com/xingchenshanyao/YOLOP-E/assets/116085226/14d4fc6a-4a72-4c39-92f8-6abb2e1b0a43)

#### Drivable Area Segmentation Result
![2](https://github.com/xingchenshanyao/YOLOP-E/assets/116085226/480ee90e-68f3-452b-b7f4-eff3feb7abac)

#### Lane Detection Result
![3](https://github.com/xingchenshanyao/YOLOP-E/assets/116085226/39ca017e-5a99-4cc0-8aa5-40ae294cc18e)

***

### Beginning


***

### Demo Test



### Demonstration
<table>
    <tr>
            <th>Output1</th>
            <th>Output2</th>
    </tr>
    <tr>
        <td><img src=inference/videos/show1.gif /></td>
        <td><img src=inference/videos/show2.gif/></td>
    </tr>
</table>

***

### Acknowledgements:

[YOLOP](https://github.com/hustvl/YOLOP)

[华夏街景](https://www.bilibili.com/video/BV1xN4y1w7pv/)
