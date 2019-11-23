Class Activation Map
==
**Unofficial Pytorch Implementation of 'Learning Deep Features for Discriminative Localization'**

**Reference**: [Learning Deep Features for Discriminative Localization, CVPR2016](https://arxiv.org/abs/1512.04150)

**Contact**: `Minseong Kim` (tyui592@gmail.com)

I used the [Networks](https://pytorch.org/docs/stable/torchvision/models.html#torchvision-models) that trained ImageNet data from `torchvision.models`.


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
* `--network`: Network for backbone (Possible networks: *resnet50, resnext50_32x4d, wide_resnet50_2, googlenet, densenet161, inception_v3, shufflenet_v2_x1_0, mobilenet_v2, mnasnet1_0*)
* `--image`: Input image path
* `--topk`: Create k Class Activation Maps (CAMs) with the highest probability
* `--imsize`: Size to resize image (maintaining aspect ratio)
* `--cropsize`: Size to crop cetenr region
* `--blend-alpha`: Interpolation factor to overlay the input with CAM 
* `--save-path`: Path to save outputs

### Example Script

`python cam.py --image imgs/input/img1.jpg --topk 3 --imsize 256 --network resnet50`

Results
--

![reulsts](imgs/results.png)

