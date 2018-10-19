"""
Modify the original file to make the class support feature extraction
"""
import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo

from ..op_wrapper import Conv2d as flexConv2d
from ..op_wrapper import BatchNorm2d as flexBatchNorm2d
from ..op_wrapper import Linear as flexLinear

class flexBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(flexBottleneck, self).__init__()
        self.conv1 = flexConv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = flexBatchNorm2d(planes)
        self.conv2 = flexConv2d(planes, planes, kernel_size=3, stride=stride,
                                padding=1, bias=False)
        self.bn2 = flexBatchNorm2d(planes)
        self.conv3 = flexConv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = flexBatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ReShadowNet2D(nn.Module):

    def __init__(self, block, layers, num_classes=1000, feat=False):
        self.inplanes = 64
        super(ReShadowNet2D, self).__init__()
        self.feat = feat
        self.conv1 = flexConv3d(3, 64, kernel_size=(1,7,7), stride=(1,2,2), padding=(0,3,3), bias=False)
        self.bn1 = flexBatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool3d((1,7,7), stride=1)
        self.feat_dim = 512 * block.expansion
        if not feat:
            self.fc = flexLinear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                flexConv3d(self.inplanes, planes * block.expansion,
                        kernel_size=1, stride=stride, bias=False),
                flexBatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if not self.feat:
            x = self.fc(x)

        return x

def resnet50_shadow(feat=False, **kwargs):
    """Constructs a ResNet-50 shadow model.
    Args:
        feat: if True, abandon pre-defined fc layer
    """
    model = ReShadowNet2D(flexBottleneck, [3, 4, 6, 3], feat=feat, **kwargs)
    return model