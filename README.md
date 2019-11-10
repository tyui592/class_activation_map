Class Activation Map
==
**This is Pytorch Implementation to generate the Class Activation Map using Resnet34**

**Reference**: [Learning Deep Features for Discriminative Localization, CVPR2016](https://arxiv.org/abs/1512.04150)

**Contact**: Minseong Kim (tyui592@gmail.com)


Requirements
--
* Pytorch (version: 1.2.0)
* Pillow (version: 6.1.0)
* matplotlib (version: 3.1.1)
* numpy (version: 1.16.5)

Usage
--

### Arguments
* `--gpu-no`: Number of gpu device (-1: cpu, 0~n: gpu)
* `--image`: Input image path
* `--topk`: Create k Class Activation Maps (CAMs) with the highest probability
* `--imsize`: Size to resize image (maintaining aspect ratio)
* `--cropsize`: Size to crop cetenr region
* `--blend-alpha`: Interpolation factor to overlay the input with CAM 
* `--save-path`: Path to save outputs

#### Script

`python cam.py --image imgs/image1.jpg --topk 3 --imsize 256`

Results
--

![figure1](https://github.com/tyui592/class_activation_map/blob/master/imgs/cam1.png)
![figure2](https://github.com/tyui592/class_activation_map/blob/master/imgs/cam2.png)
![figure3](https://github.com/tyui592/class_activation_map/blob/master/imgs/cam3.png)
![figure4](https://github.com/tyui592/class_activation_map/blob/master/imgs/cam4.png)

