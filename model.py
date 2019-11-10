import torch
import torch.nn as nn

import torchvision

class CAM(nn.Module):
    def __init__(self, network='resnet34'):
        super(CAM, self).__init__()
        
        net = torchvision.models.__dict__[network](pretrained=True)        
        netList = list(net.children())
        
        self.feature_extractor = nn.Sequential(*netList[:-2])
        self.gap = netList[-2]        
        self.fully_connected_layer = netList[-1]
        
    def forward(self, x, topk=3):
        target_size = x.size(2), x.size(3)
            
        # extract feature map
        fmap = self.feature_extractor(x)
        
        # Classify the input image
        flatten = self.gap(fmap)
        output = self.fully_connected_layer(flatten.view(flatten.size(0), -1))
        output = nn.functional.softmax(output, dim=1)
        prob, args = torch.sort(output, dim=1, descending=True)
        
        ## top k class probability
        topk_prob = prob.squeeze().tolist()[:topk]
        topk_arg = args.squeeze().tolist()[:topk]
        
        # Generage class activation map
        b, c, h, w = fmap.size()
        fmap = fmap.view(b, c, h*w).transpose(1, 2)

        fc_weight = self.fully_connected_layer.weight.t().unsqueeze(0)        
        cam = torch.bmm(fmap, fc_weight).transpose(1, 2)

        ## normalize to 0 ~ 1
        min_val, min_args = torch.min(cam, dim=2, keepdim=True)
        cam -= min_val
        max_val, max_args = torch.max(cam, dim=2, keepdim=True)
        cam /= max_val
        
        ## top k class activation map
        topk_cam = cam.view(1, -1, h, w)[0, topk_arg]
        topk_cam = nn.functional.interpolate(topk_cam.unsqueeze(0), 
                                        target_size, mode='bilinear', align_corners=True).squeeze()
        topk_cam = torch.split(topk_cam, 1)
        
        return topk_prob, topk_arg, topk_cam
