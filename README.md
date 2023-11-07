## YOLOP-E: You Only Look Once for Expressway Panoramic Driving Perception

Our paper: YOLOP-E: You Only Look Once for Expressway Panoramic Driving Perception(Submitted)

Our datasets and weights: [SDExpressway, yolop.pth, yolope.pth](https://pan.baidu.com/s/1589VGpmHATSrTs6f_HSI_g?pwd=m2jh)

### The Illustration of YOLOP-E
![图片1](https://github.com/xingchenshanyao/YOLOP-E/assets/116085226/5588ffc2-4b3e-40e2-92a5-429ec596c03b)

### The Illustration of ELAN-X

### Contributions
* This study has produced the expressway multi-task dataset, SDExpressway, encompassing 5603 images captured under various weather conditions, including sunny, dark, rainy and foggy scenarios. Each image in the dataset has been meticulously labeled with drivable areas, lane lines, and traffic object information.

* This research endeavors include the optimization of the ELAN module, resulting in the creation of a more efficient aggregated network structure known as ELAN-X. This innovation facilitates the amalgamation of feature information from various depths for parallel processing, enhancing the sensory field and feature expression of the model. These enhancements bolster the accuracy of the multi-task model's detection capabilities.

* This paper introduce an efficient multi-task network, YOLOP-E, tailored for expressway scenarios and built upon the YOLOP framework. YOLOP-E is engineered to jointly handle three critical tasks in autonomous driving: traffic object detection, drivable area segmentation, and lane line segmentation.

*  The proposed network undergoes extensive evaluation on both the SDExpressway dataset and the widely recognized BDD100k dataset, including ablation experiments and state-of-the-art (SOTA) comparison experiments to demonstrate the efficacy of the various improvements integrated into the model. Notably, the proposed model showcases robustness and strong generalization abilities, even in challenging environmental conditions.

### Results

#### On the SDExpressway

##### Traffic Object Detection Result
| Network          | R(%) | mAP50(%) | mAP50:95(%) | FPS(fps) |
| :--------------: | :---------: | :--------: | :----------: | :----------: |
| `YOLOP(baseline)`     | 86.8      | 74.4     | 38.7     | 232        |
| `HybridNets`      | 90.1      | 76.4     | 42.1     | 110        |
| `YOLOP-E(ours)`  | 92.1      | 83.8     | 53.3     | 127         |
<table class="MsoTableGrid" border="1" cellspacing="0" style="border-collapse:collapse;border:none;mso-border-left-alt:0.5000pt solid windowtext;
mso-border-top-alt:0.5000pt solid windowtext;mso-border-right-alt:0.5000pt solid windowtext;mso-border-bottom-alt:0.5000pt solid windowtext;
mso-border-insideh:0.5000pt solid windowtext;mso-border-insidev:0.5000pt solid windowtext;mso-padding-alt:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;"><tbody><tr><td width="74" valign="top" rowspan="2" style="width:44.8500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:1.0000pt solid rgb(0,0,0);mso-border-top-alt:0.5000pt solid rgb(0,0,0);border-bottom:1.0000pt solid windowtext;
mso-border-bottom-alt:0.5000pt solid windowtext;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;">Class</span><span style="font-family:Calibri;mso-fareast-font-family:宋体;mso-bidi-font-family:'Times New Roman';
font-size:10.5000pt;mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="212" valign="top" colspan="3" style="width:127.6500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:1.0000pt solid rgb(0,0,0);mso-border-top-alt:0.5000pt solid rgb(0,0,0);border-bottom:1.0000pt solid rgb(0,0,0);
mso-border-bottom-alt:0.5000pt solid rgb(0,0,0);"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;">YOLOP</span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="211" valign="top" colspan="3" style="width:126.8000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:1.0000pt solid rgb(0,0,0);mso-border-top-alt:0.5000pt solid rgb(0,0,0);border-bottom:1.0000pt solid rgb(0,0,0);
mso-border-bottom-alt:0.5000pt solid rgb(0,0,0);"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;">HybridNets</span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="211" valign="top" colspan="3" style="width:126.8000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:1.0000pt solid rgb(0,0,0);mso-border-top-alt:0.5000pt solid rgb(0,0,0);border-bottom:1.0000pt solid rgb(0,0,0);
mso-border-bottom-alt:0.5000pt solid rgb(0,0,0);"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;">YOLOP</span><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">-</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;">E</span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td></tr><tr><td width="46" valign="top" style="width:27.9000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:0.5000pt solid rgb(0,0,0);border-bottom:1.0000pt solid rgb(0,0,0);
mso-border-bottom-alt:0.5000pt solid rgb(0,0,0);"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;">R</span><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">(%)</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="75" valign="top" style="width:45.0000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:0.5000pt solid rgb(0,0,0);border-bottom:1.0000pt solid rgb(0,0,0);
mso-border-bottom-alt:0.5000pt solid rgb(0,0,0);"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;">mAP50</span><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">(%)</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="91" valign="top" style="width:54.7500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:0.5000pt solid rgb(0,0,0);border-bottom:1.0000pt solid rgb(0,0,0);
mso-border-bottom-alt:0.5000pt solid rgb(0,0,0);"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;">mAP50:95</span><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">(%)</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="45" valign="top" style="width:27.0500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:1.0000pt solid rgb(0,0,0);mso-border-top-alt:0.5000pt solid rgb(0,0,0);border-bottom:1.0000pt solid rgb(0,0,0);
mso-border-bottom-alt:0.5000pt solid rgb(0,0,0);"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;">R</span><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">(%)</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="75" valign="top" style="width:45.0000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:1.0000pt solid rgb(0,0,0);mso-border-top-alt:0.5000pt solid rgb(0,0,0);border-bottom:1.0000pt solid rgb(0,0,0);
mso-border-bottom-alt:0.5000pt solid rgb(0,0,0);"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;">mAP50</span><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">(%)</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="91" valign="top" style="width:54.7500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:1.0000pt solid rgb(0,0,0);mso-border-top-alt:0.5000pt solid rgb(0,0,0);border-bottom:1.0000pt solid rgb(0,0,0);
mso-border-bottom-alt:0.5000pt solid rgb(0,0,0);"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;">mAP50:95</span><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">(%)</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="45" valign="top" style="width:27.0500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:1.0000pt solid rgb(0,0,0);mso-border-top-alt:0.5000pt solid rgb(0,0,0);border-bottom:1.0000pt solid rgb(0,0,0);
mso-border-bottom-alt:0.5000pt solid rgb(0,0,0);"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;">R</span><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">(%)</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="75" valign="top" style="width:45.0000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:1.0000pt solid rgb(0,0,0);mso-border-top-alt:0.5000pt solid rgb(0,0,0);border-bottom:1.0000pt solid rgb(0,0,0);
mso-border-bottom-alt:0.5000pt solid rgb(0,0,0);"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;">mAP50</span><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">(%)</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="91" valign="top" style="width:54.7500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:1.0000pt solid rgb(0,0,0);mso-border-top-alt:0.5000pt solid rgb(0,0,0);border-bottom:1.0000pt solid rgb(0,0,0);
mso-border-bottom-alt:0.5000pt solid rgb(0,0,0);"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;">mAP50:95</span><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">(%)</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td></tr><tr><td width="74" valign="top" style="width:44.8500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:0.5000pt solid rgb(0,0,0);border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;">All</span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="46" valign="top" style="width:27.9000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:0.5000pt solid rgb(0,0,0);border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><b><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-weight:bold;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><font face="Times New Roman">86.7</font></span></b><b><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-weight:bold;
font-size:7.5000pt;mso-font-kerning:1.0000pt;"><o:p></o:p></span></b></p></td><td width="75" valign="top" style="width:45.0000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:0.5000pt solid rgb(0,0,0);border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><b><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-weight:bold;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><font face="Times New Roman">74.4</font></span></b><b><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-weight:bold;
font-size:7.5000pt;mso-font-kerning:1.0000pt;"><o:p></o:p></span></b></p></td><td width="91" valign="top" style="width:54.7500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:0.5000pt solid rgb(0,0,0);border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><b><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-weight:bold;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><font face="Times New Roman">38.7</font></span></b><b><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-weight:bold;
font-size:7.5000pt;mso-font-kerning:1.0000pt;"><o:p></o:p></span></b></p></td><td width="45" valign="top" style="width:27.0500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:0.5000pt solid rgb(0,0,0);border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><b><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-weight:bold;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><font face="Times New Roman">90.1</font></span></b><b><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-weight:bold;
font-size:7.5000pt;mso-font-kerning:1.0000pt;"><o:p></o:p></span></b></p></td><td width="75" valign="top" style="width:45.0000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:0.5000pt solid rgb(0,0,0);border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><b><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-weight:bold;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><font face="Times New Roman">76.4</font></span></b><b><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-weight:bold;
font-size:7.5000pt;mso-font-kerning:1.0000pt;"><o:p></o:p></span></b></p></td><td width="91" valign="top" style="width:54.7500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:0.5000pt solid rgb(0,0,0);border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><b><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-weight:bold;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><font face="Times New Roman">42.1</font></span></b><b><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-weight:bold;
font-size:7.5000pt;mso-font-kerning:1.0000pt;"><o:p></o:p></span></b></p></td><td width="45" valign="top" style="width:27.0500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:0.5000pt solid rgb(0,0,0);border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><b><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-weight:bold;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><font face="Times New Roman">92.1</font></span></b><b><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-weight:bold;
font-size:7.5000pt;mso-font-kerning:1.0000pt;"><o:p></o:p></span></b></p></td><td width="75" valign="top" style="width:45.0000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:0.5000pt solid rgb(0,0,0);border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><b><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-weight:bold;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><font face="Times New Roman">83.8</font></span></b><b><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-weight:bold;
font-size:7.5000pt;mso-font-kerning:1.0000pt;"><o:p></o:p></span></b></p></td><td width="91" valign="top" style="width:54.7500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:0.5000pt solid rgb(0,0,0);border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><b><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-weight:bold;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><font face="Times New Roman">53.3</font></span></b><b><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-weight:bold;
font-size:7.5000pt;mso-font-kerning:1.0000pt;"><o:p></o:p></span></b></p></td></tr><tr><td width="74" valign="top" style="width:44.8500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;">Car</span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="46" valign="top" style="width:27.9000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">93.0</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="75" valign="top" style="width:45.0000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">90.1</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="91" valign="top" style="width:54.7500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">53.4</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="45" valign="top" style="width:27.0500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">96.3</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="75" valign="top" style="width:45.0000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">90.5</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="91" valign="top" style="width:54.7500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">58.4</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="45" valign="top" style="width:27.0500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">94.0</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="75" valign="top" style="width:45.0000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">92.3</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="91" valign="top" style="width:54.7500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">62.6</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td></tr><tr><td width="74" valign="top" style="width:44.8500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;">Truck</span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="46" valign="top" style="width:27.9000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">97.8</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="75" valign="top" style="width:45.0000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">95.5</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="91" valign="top" style="width:54.7500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">62.9</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="45" valign="top" style="width:27.0500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">98.9</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="75" valign="top" style="width:45.0000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">95.7</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="91" valign="top" style="width:54.7500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">64.9</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="45" valign="top" style="width:27.0500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">98.7</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="75" valign="top" style="width:45.0000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">97.1</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="91" valign="top" style="width:54.7500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">73.2</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td></tr><tr><td width="74" valign="top" style="width:44.8500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;">G</span><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">.</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><span style="mso-spacerun:'yes';">&nbsp;</span>Sign</span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="46" valign="top" style="width:27.9000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">95.2</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="75" valign="top" style="width:45.0000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">86.9</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="91" valign="top" style="width:54.7500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">52.0</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="45" valign="top" style="width:27.0500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">96.9</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="75" valign="top" style="width:45.0000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">81.9</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="91" valign="top" style="width:54.7500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">53.0</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="45" valign="top" style="width:27.0500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">98.6</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="75" valign="top" style="width:45.0000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">92.3</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="91" valign="top" style="width:54.7500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">66.5</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td></tr><tr><td width="74" valign="top" style="width:44.8500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;">D</span><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">.</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><span style="mso-spacerun:'yes';">&nbsp;</span>Sign</span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="46" valign="top" style="width:27.9000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">88.3</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="75" valign="top" style="width:45.0000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">69.1</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="91" valign="top" style="width:54.7500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">30.1</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="45" valign="top" style="width:27.0500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">93.7</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="75" valign="top" style="width:45.0000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">71.9</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="91" valign="top" style="width:54.7500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">35.9</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="45" valign="top" style="width:27.0500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">100</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="75" valign="top" style="width:45.0000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">87.3</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="91" valign="top" style="width:54.7500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">40.0</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td></tr><tr><td width="74" valign="top" style="width:44.8500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;">W</span><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">.</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><span style="mso-spacerun:'yes';">&nbsp;</span>Sign</span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="46" valign="top" style="width:27.9000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">84.7</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="75" valign="top" style="width:45.0000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">66.4</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="91" valign="top" style="width:54.7500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">30.9</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="45" valign="top" style="width:27.0500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">93.7</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="75" valign="top" style="width:45.0000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">71.2</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="91" valign="top" style="width:54.7500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">34.5</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="45" valign="top" style="width:27.0500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">93.1</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="75" valign="top" style="width:45.0000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">86.8</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="91" valign="top" style="width:54.7500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">51.4</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td></tr><tr><td width="74" valign="top" style="width:44.8500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;">E</span><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">.</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;">T</span><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">.</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><span style="mso-spacerun:'yes';">&nbsp;</span>Sign</span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="46" valign="top" style="width:27.9000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">75.0</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="75" valign="top" style="width:45.0000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">65.4</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="91" valign="top" style="width:54.7500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">27.3</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="45" valign="top" style="width:27.0500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">75.5</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="75" valign="top" style="width:45.0000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">69.6</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="91" valign="top" style="width:54.7500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">30.7</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="45" valign="top" style="width:27.0500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">87.5</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="75" valign="top" style="width:45.0000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">79.3</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="91" valign="top" style="width:54.7500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">62.5</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td></tr><tr><td width="74" valign="top" style="width:44.8500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;">S</span><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">.</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;">L</span><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">. </font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;">Sign</span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="46" valign="top" style="width:27.9000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">82.1</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="75" valign="top" style="width:45.0000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">61.8</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="91" valign="top" style="width:54.7500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">23.8</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="45" valign="top" style="width:27.0500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">86.2</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="75" valign="top" style="width:45.0000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">75.0</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="91" valign="top" style="width:54.7500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">35.2</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="45" valign="top" style="width:27.0500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">94.6</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="75" valign="top" style="width:45.0000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">83.7</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="91" valign="top" style="width:54.7500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">37.2</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td></tr><tr><td width="74" valign="top" style="width:44.8500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;">P</span><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">.</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><span style="mso-spacerun:'yes';">&nbsp;</span>Sign</span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="46" valign="top" style="width:27.9000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">94.4</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="75" valign="top" style="width:45.0000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">87.9</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="91" valign="top" style="width:54.7500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">54.0</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="45" valign="top" style="width:27.0500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">96.5</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="75" valign="top" style="width:45.0000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">84.4</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="91" valign="top" style="width:54.7500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">54.9</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="45" valign="top" style="width:27.0500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">94.4</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="75" valign="top" style="width:45.0000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">87.5</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="91" valign="top" style="width:54.7500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">66.8</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td></tr><tr><td width="74" valign="top" style="width:44.8500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">S. </font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;">Arrow</span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="46" valign="top" style="width:27.9000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">78.4</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="75" valign="top" style="width:45.0000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">62.6</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="91" valign="top" style="width:54.7500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">28.4</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="45" valign="top" style="width:27.0500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">94.3</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="75" valign="top" style="width:45.0000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">68.9</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="91" valign="top" style="width:54.7500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">31.1</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="45" valign="top" style="width:27.0500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">81.8</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="75" valign="top" style="width:45.0000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">73.7</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="91" valign="top" style="width:54.7500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:none;
mso-border-bottom-alt:none;"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">46.0</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td></tr><tr><td width="74" valign="top" style="width:44.8500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:1.0000pt solid rgb(0,0,0);
mso-border-bottom-alt:0.5000pt solid rgb(0,0,0);"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;">S</span><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">.</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;">R</span><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">. </font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;">Arrow</span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="46" valign="top" style="width:27.9000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:1.0000pt solid rgb(0,0,0);
mso-border-bottom-alt:0.5000pt solid rgb(0,0,0);"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">77.8</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="75" valign="top" style="width:45.0000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:1.0000pt solid rgb(0,0,0);
mso-border-bottom-alt:0.5000pt solid rgb(0,0,0);"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">56.9</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="91" valign="top" style="width:54.7500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:1.0000pt solid rgb(0,0,0);
mso-border-bottom-alt:0.5000pt solid rgb(0,0,0);"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">23.8</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="45" valign="top" style="width:27.0500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:1.0000pt solid rgb(0,0,0);
mso-border-bottom-alt:0.5000pt solid rgb(0,0,0);"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">68.6</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="75" valign="top" style="width:45.0000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:1.0000pt solid rgb(0,0,0);
mso-border-bottom-alt:0.5000pt solid rgb(0,0,0);"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">55.0</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="91" valign="top" style="width:54.7500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:1.0000pt solid rgb(0,0,0);
mso-border-bottom-alt:0.5000pt solid rgb(0,0,0);"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">22.3</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="45" valign="top" style="width:27.0500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:1.0000pt solid rgb(0,0,0);
mso-border-bottom-alt:0.5000pt solid rgb(0,0,0);"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">77.8</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="75" valign="top" style="width:45.0000pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:1.0000pt solid rgb(0,0,0);
mso-border-bottom-alt:0.5000pt solid rgb(0,0,0);"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">57.8</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td><td width="91" valign="top" style="width:54.7500pt;padding:0.0000pt 5.4000pt 0.0000pt 5.4000pt ;border-left:none;
mso-border-left-alt:none;border-right:none;mso-border-right-alt:none;
border-top:none;mso-border-top-alt:none;border-bottom:1.0000pt solid rgb(0,0,0);
mso-border-bottom-alt:0.5000pt solid rgb(0,0,0);"><p class="MsoNormal" align="center" style="text-align:center;"><span style="font-family:宋体;mso-ascii-font-family:'Times New Roman';mso-hansi-font-family:'Times New Roman';
mso-bidi-font-family:'Times New Roman';font-size:7.5000pt;mso-font-kerning:1.0000pt;"><font face="Times New Roman">26.5</font></span><span style="font-family:'Times New Roman';mso-fareast-font-family:宋体;font-size:7.5000pt;
mso-font-kerning:1.0000pt;"><o:p></o:p></span></p></td></tr></tbody></table>

##### Drivable Area Segmentation Result
| Network          | mIoU(%) | FPS(fps) |
| :--------------: | :---------: | :--------: |
| `YOLOP(baseline)`     | 97.7      | 232     |
| `HybridNets`      | 97.5      | 110     |
| `YOLOP-E(ours)`  | 98.1      | 127     |
##### Lane Detection Result
| Network          | Acc(%) |IoU(fps) | FPS(fps) |
| :--------------: | :---------: | :--------: | :--------: |
| `YOLOP(baseline)`     | 90.8      | 72.8     | 232     |
| `HybridNets`      | 92.0      | 75.7     | 110    |
| `YOLOP-E(ours)`  | 92.1      | 76.2     | 127     |

##### On the BDD100k

##### Traffic Object Detection Result
| Network          | R(%) | mAP50(%) | mAP50:95(%) | FPS(fps) |
| :--------------: | :---------: | :--------: | :----------: | :----------: |
| `YOLOP(baseline)`     | 89.5      | 76.3     | 43.1     | 230        |
| `MultiNet`      | 81.3      | 60.2     | 33.1     | 51        |
| `DLT-Net`  | 89.4      | 68.4     | 38.6     | 56         |
| `HybridNets`      | 92.8      | 77.3     | 45.8     | 108        |
| `YOLOP-E(ours)`  | 92.0      | 79.7     | 46.7     | 120         |
##### Drivable Area Segmentation Result
| Network          | mIoU(%) | FPS(fps) |
| :--------------: | :---------: | :--------: |
| `YOLOP(baseline)`     | 91.3      | 230     |
| `MultiNet`      | 71.6     | 51     |
| `DLT-Net`  | 71.3     | 56     |
| `HybridNets`      | 90.5      | 108     |
| `YOLOP-E(ours)`  | 92.1      | 120     |
##### Lane Detection Result
| Network          | Acc(%) |IoU(fps) | FPS(fps) |
| :--------------: | :---------: | :--------: | :--------: |
| `YOLOP(baseline)`     | 70.5      | 26.2    | 230     |
| `HybridNets`      | 85.4      | 31.6     | 108   |
| `YOLOP-E(ours)`  | 73.0      | 27.3     | 120     |

### Visualization

#### Traffic Object Detection Result
![1](https://github.com/xingchenshanyao/YOLOP-E/assets/116085226/eb4e0e3a-296f-4484-aedc-07c362cbd6fb)
#### Drivable Area Segmentation Result
![2](https://github.com/xingchenshanyao/YOLOP-E/assets/116085226/33071530-7695-4cb5-b91a-77411c032dcf)
#### Lane Detection Result
![3](https://github.com/xingchenshanyao/YOLOP-E/assets/116085226/7a4909a2-b70b-4161-a4d9-0b8a62ef7d4f)

### Demonstration
<table>
    <tr>
            <th>show1</th>
            <th>show2</th>
    </tr>
    <tr>
        <td><img src=inference/videos/show1.gif /></td>
        <td><img src=inference/videos/show2.gif/></td>
    </tr>
</table>

### Acknowledgements:

[YOLOP](https://github.com/hustvl/YOLOP)

[华夏街景](https://www.bilibili.com/video/BV1xN4y1w7pv/)
