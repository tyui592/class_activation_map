Class Activation Map
==
**This is Pytorch Implementation to generate the Class Activation Map using Resnet34**

**Reference**: [Learning Deep Features for Discriminative Localization, CVPR2016](https://arxiv.org/abs/1512.04150)

**Contact**: `Minseong Kim` (tyui592@gmail.com)

I used the pre-trained [Networks](https://pytorch.org/docs/stable/torchvision/models.html#torchvision-models) from `torchvision.models`.


Requirements
--
* torch (version: 1.2.0)
* torchvision (version: 0.4.0)
* Pillow (version: 6.1.0)
* matplotlib (version: 3.1.1)
* numpy (version: 1.16.5)

Usage
--

### Arguments
* `--gpu-no`: Number of gpu device (-1: cpu, 0~n: gpu)
* `--network`: Network for backbone (Possible networks: resnet50, resnext50_32x4d, wide_resnet50_2, googlenet, densenet161, inception_v3, shufflenet_v2_x1_0, mobilenet_v2, mnasnet1_0)
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

