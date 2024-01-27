import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34
from torchvision.transforms import InterpolationMode
from collections import OrderedDict


def BNReLU(num_features):
    return nn.Sequential(
            nn.BatchNorm2d(num_features),
            nn.ReLU())

# Context Semantic Branch
class Semantic_Branch(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        resnet = resnet18(pretrained); resnet.layer3 = nn.Identity()
        resnet.layer4 = nn.Identity(); resnet.avgpool = nn.Identity()
        resnet.fc = nn.Identity(); 
        self.model = nn.Sequential(OrderedDict([("prefix", nn.Sequential(resnet.conv1,
                                                                        resnet.bn1,
                                                                        resnet.relu,
                                                                        resnet.maxpool)),
                                                ("layer1", resnet.layer1),
                                                ("layer2", resnet.layer2),
                                                ]))
    def forward(self, x):
        x = self.model.prefix(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        return x

# Spatial Detail Branch
class Detail_Branch(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = resnet18(pretrained=True); resnet.layer2 = nn.Identity()
        resnet.layer3 = nn.Identity(); resnet.layer4 = nn.Identity()
        resnet.avgpool = nn.Identity(); resnet.fc = nn.Identity()
        self.model = nn.Sequential(OrderedDict([("prefix", nn.Sequential(resnet.conv1,
                                                                        resnet.bn1,
                                                                        resnet.relu,
                                                                        resnet.maxpool)),
                                                ("layer1", resnet.layer1),
                                                ]))

        self.channel_adjust = nn.Sequential(nn.Conv2d(64, 128, 1), BNReLU(128))

    def forward(self, x):
        x = self.model(x)
        x = self.channel_adjust(x)
        return x

# selective fusion module
class Fusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.point_conv = nn.Sequential(nn.Conv2d(256, 128, 1),
                                    BNReLU(128),
                                    nn.Conv2d(128, 1, 1))

    def forward(self, low, high):
        high = F.interpolate(high, size=(low.size(2), low.size(3)),
                                    mode="bilinear", align_corners=False)
        attmap = torch.cat([high, low], dim=1)
        attmap = self.point_conv(attmap)
        attmap = torch.sigmoid(attmap)
        return attmap * low + high

# core structure of LFD-RoadSeg 
class LFD_RoadSeg(nn.Module):
    def __init__(self, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor

        self.semantic = Semantic_Branch(pretrained=True) 

        self.context1 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=(1, 5), dilation=2, padding=(0, 4), groups=128),
                                        nn.Conv2d(128, 128, kernel_size=(5, 1), dilation=1, padding=(2, 0),  groups=128),
                                        nn.Conv2d(128, 128, kernel_size=1),
                                        BNReLU(128))

        self.context2 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=(1, 5), dilation=1, padding=(0, 2), groups=128), 
                                        nn.Conv2d(128, 128, kernel_size=(5, 1), dilation=1, padding=(2, 0),  groups=128),
                                        nn.Conv2d(128, 128, kernel_size=1),
                                        BNReLU(128))

        self.detail = Detail_Branch() 
        self.fusion = Fusion()

        self.cls_head = nn.Sequential(nn.Conv2d(128, 128, kernel_size=1),
                                        BNReLU(128),
                                        nn.Conv2d(128, 2, kernel_size=1))

    def forward(self, data):
        """
            Args:
                data: A dictionary containing "img", "label", and "filename"
                data["img"]: (b, c, h, w)
        """
        x_ = data["img"]
        x__ = F.interpolate(x_, size=(x_.size(2)//self.scale_factor, x_.size(3)//(2*self.scale_factor)), 
                        mode="bilinear", align_corners=False) 
    
        # context semantic branch
        x_1 = self.semantic(x__)
        del x__

        # aggregation module
        x_1 = self.context1(x_1) + x_1
        x_1 = self.context2(x_1) + x_1

        # spatial detail branch
        x = self.detail(x_)

        # selective fusion module
        x_1 = self.fusion(x,x_1)
        
        score_map = self.cls_head(x_1)
        score_map = F.interpolate(score_map, size=(x_.size(2), x_.size(3)), 
                        mode="bilinear", align_corners=False)
        
        return score_map