from typing import Callable, Any, Optional

import torch
from torch import Tensor
from torch import nn
import torchvision


class MobileNetV1(nn.Module):
    def __init__(self, num_classes=10, norm_layer=None):
        super().__init__()

        def conv_bn(inp, oup, stride, norm_layer):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                norm_layer(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride, norm_layer):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                norm_layer(inp),
                nn.ReLU(inplace=True),
    
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                norm_layer(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(  3,  32, 2, norm_layer), 
            conv_dw( 32,  64, 1, norm_layer),
            conv_dw( 64, 128, 2, norm_layer),
            conv_dw(128, 128, 1, norm_layer),
            conv_dw(128, 256, 2, norm_layer),
            conv_dw(256, 256, 1, norm_layer),
            conv_dw(256, 512, 2, norm_layer),
            conv_dw(512, 512, 1, norm_layer),
            conv_dw(512, 512, 1, norm_layer),
            conv_dw(512, 512, 1, norm_layer),
            conv_dw(512, 512, 1, norm_layer),
            conv_dw(512, 512, 1, norm_layer),
            conv_dw(512, 1024, 2, norm_layer),
            conv_dw(1024, 1024, 1, norm_layer),
            # nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x



def mobilenet_v1(num_classes, norm_layer) -> MobileNetV1:
    model = MobileNetV1(num_classes=num_classes, norm_layer=norm_layer)

    return model