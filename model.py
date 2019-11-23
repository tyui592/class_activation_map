import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision


class CAM(nn.Module):
    def __init__(self, network='resnet50'):
        super(CAM, self).__init__()

        if network in ['resnet50', 'resnext50_32x4d', 'wide_resnet50_2']:
            self.network = ResNet(network)

        elif network in ['googlenet']:
            self.network = GoogleNet(network)

        elif network in ['densenet161']:
            self.network = DensetNet(network)

        elif network in ['inception_v3']:
            self.network = InceptionNet(network)

        elif network in ['shufflenet_v2_x1_0']:
            self.network = ShuffleNet(network)

        elif network in ['mnasnet1_0']:
            self.network = MnasNet(network)

        elif network in ['mobilenet_v2']:
            self.network = MobileNet(network)
        else:
            raise NotImplementedError("Not expected network")
        
        
    def forward(self, x, topk=3):
        feature_map, output = self.network(x)
        prob, args = torch.sort(output, dim=1, descending=True)
        
        ## top k class probability
        topk_prob = prob.squeeze().tolist()[:topk]
        topk_arg = args.squeeze().tolist()[:topk]
        
        # generage class activation map
        b, c, h, w = feature_map.size()
        feature_map = feature_map.view(b, c, h*w).transpose(1, 2)

        cam = torch.bmm(feature_map, self.network.fc_weight).transpose(1, 2)

        ## normalize to 0 ~ 1
        min_val, min_args = torch.min(cam, dim=2, keepdim=True)
        cam -= min_val
        max_val, max_args = torch.max(cam, dim=2, keepdim=True)
        cam /= max_val
        
        ## top k class activation map
        topk_cam = cam.view(1, -1, h, w)[0, topk_arg]
        topk_cam = nn.functional.interpolate(topk_cam.unsqueeze(0), 
                                        (x.size(2), x.size(3)), mode='bilinear', align_corners=True).squeeze(0)
        topk_cam = torch.split(topk_cam, 1)

        return topk_prob, topk_arg, topk_cam

class ResNet(nn.Module):
    def __init__(self, network):
        super(ResNet ,self).__init__()
        net = torchvision.models.__dict__[network](pretrained=True)        
        net_list = list(net.children())
        
        self.feature_extractor = nn.Sequential(*net_list[:-2])
        self.fc_layer = net_list[-1]
        self.fc_weight = nn.Parameter(self.fc_layer.weight.t().unsqueeze(0))

    def forward(self, x):
        feature_map = self.feature_extractor(x)
        output = F.softmax(self.fc_layer(feature_map.mean([2, 3])), dim=1)
        return feature_map, output

class GoogleNet(nn.Module):
    def __init__(self, network):
        super(GoogleNet, self).__init__()
        net = torchvision.models.__dict__[network](pretrained=True)
        net_list = list(net.children())

        self.feature_extractor = nn.Sequential(*net_list[:-3])
        self.fc_layer = net_list[-1]
        self.fc_weight = nn.Parameter(self.fc_layer.weight.t().unsqueeze(0))

    def forward(self, x):
        feature_map = self.feature_extractor(x)
        output = F.softmax(self.fc_layer(feature_map.mean([2, 3])), dim=1)
        return feature_map, output

class DensetNet(nn.Module):
    def __init__(self, network):
        super(DensetNet, self).__init__()
        net = torchvision.models.__dict__[network](pretrained=True)

        self.feature_extractor = net.features
        self.fc_layer = net.classifier
        self.fc_weight = nn.Parameter(self.fc_layer.weight.t().unsqueeze(0))

    def forward(self, x):
        feature_map = F.relu(self.feature_extractor(x))
        output = F.softmax(self.fc_layer(feature_map.mean([2, 3])), dim=1)
        return feature_map, output

class InceptionNet(nn.Module):
    def __init__(self, network):
        super(InceptionNet, self).__init__()
        self.net = torchvision.models.__dict__[network](pretrained=True)

        self.fc_layer = self.net.fc
        self.fc_weight = nn.Parameter(self.fc_layer.weight.t().unsqueeze(0))

    def forward(self, x):
        feature_map = self._feature_extraction(x)
        output = F.softmax(self.fc_layer(feature_map.mean([2, 3])), dim=1)
        return feature_map, output

    def _feature_extraction(self, x):
        x = self.net.Conv2d_1a_3x3(x)
        x = self.net.Conv2d_2a_3x3(x)
        x = self.net.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.net.Conv2d_3b_1x1(x)
        x = self.net.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.net.Mixed_5b(x)
        x = self.net.Mixed_5c(x)
        x = self.net.Mixed_5d(x)
        x = self.net.Mixed_6a(x)
        x = self.net.Mixed_6b(x)
        x = self.net.Mixed_6c(x)
        x = self.net.Mixed_6d(x)
        x = self.net.Mixed_6e(x)
        x = self.net.Mixed_7a(x)
        x = self.net.Mixed_7b(x)
        x = self.net.Mixed_7c(x)
        return x

class ShuffleNet(nn.Module):
    def __init__(self, network):
        super(ShuffleNet, self).__init__()
        self.net = torchvision.models.__dict__[network](pretrained=True)

        self.fc_layer = self.net.fc
        self.fc_weight = nn.Parameter(self.fc_layer.weight.t().unsqueeze(0))

    def forward(self, x):
        feature_map = self._feature_extraction(x)
        output = F.softmax(self.fc_layer(feature_map.mean([2, 3])), dim=1)
        return feature_map, output

    def _feature_extraction(self, x):
        x = self.net.conv1(x)
        x = self.net.maxpool(x)
        x = self.net.stage2(x)
        x = self.net.stage3(x)
        x = self.net.stage4(x)
        x = self.net.conv5(x)
        return x

class MobileNet(nn.Module):
    def __init__(self, network):
        super(MobileNet, self).__init__()
        net = torchvision.models.__dict__[network](pretrained=True)

        self.feature_extractor = net.features
        self.fc_layer = net.classifier[-1]
        self.fc_weight = nn.Parameter(self.fc_layer.weight.t().unsqueeze(0))

    def forward(self, x):
        feature_map = self.feature_extractor(x)
        output = F.softmax(self.fc_layer(feature_map.mean([2, 3])), dim=1)
        return feature_map, output

class MnasNet(nn.Module):
    def __init__(self, network):
        super(MnasNet, self).__init__()
        net = torchvision.models.__dict__[network](pretrained=True)

        self.feature_extractor = net.layers
        self.fc_layer = net.classifier[-1]
        self.fc_weight = nn.Parameter(self.fc_layer.weight.t().unsqueeze(0))

    def forward(self, x):
        feature_map = self.feature_extractor(x)
        output = F.softmax(self.fc_layer(feature_map.mean([2, 3])), dim=1)
        return feature_map, output
