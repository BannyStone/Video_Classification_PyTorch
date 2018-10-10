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

def flex_conv3x3(in_planes, out_planes, stride=1, parameters=None):
    """3x3 convolution with padding"""
    return flexConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, parameters=parameters)

class flexBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, parameters, stride=1, downsample=None):
        super(flexBottleneck, self).__init__()
        self.conv1 = flexConv2d(inplanes, planes, kernel_size=1, bias=False, 
                                parameters=parameters['conv1'])
        self.bn1 = flexBatchNorm2d(planes, 
                                parameters=parameters['bn1'])
        self.conv2 = flexConv2d(planes, planes, kernel_size=3, stride=stride,
                                padding=1, bias=False, 
                                parameters=parameters['conv2'])
        self.bn2 = flexBatchNorm2d(planes, 
                                parameters=parameters['bn2'])
        self.conv3 = flexConv2d(planes, planes * self.expansion, kernel_size=1, bias=False, 
                                parameters=parameters['conv3'])
        self.bn3 = flexBatchNorm2d(planes * self.expansion,
                                parameters=parameters['bn3'])
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

    def __init__(self, block, layers, parameters, num_classes=1000, feat=False):
        self.inplanes = 64
        super(ReShadowNet2D, self).__init__()
        self.feat = feat
        self.conv1 = flexConv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False, 
                                parameters=parameters['conv1'])
        self.bn1 = flexBatchNorm2d(64,
                                parameters=parameters['bn1'])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], parameters['layer1'])
        self.layer2 = self._make_layer(block, 128, layers[1], parameters['layer2'], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], parameters['layer3'], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], parameters['layer4'], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.feat_dim = 512 * block.expansion
        if not feat:
            self.fc = flexLinear(512 * block.expansion, num_classes,
                                parameters=parameters['fc']) # TODO: key name may be changed

    def _make_layer(self, block, planes, blocks, parameters, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            params = parameters['downsample']
            downsample = nn.Sequential(
                flexConv2d(self.inplanes, planes * block.expansion,
                        kernel_size=1, stride=stride, bias=False,
                        parameters=params['0']),
                flexBatchNorm2d(planes * block.expansion,
                        parameters=params['1']),
            )

        layers = []
        layers.append(block(self.inplanes, planes, parameters['0'], stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, parameters[str(i)]))

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

def resnet50_shadow(parameters, feat=False, **kwargs):
    """Constructs a ResNet-50 shadow model.
    Args:
        parameters: A dictionary containing all the parameter tensors in this model.
        feat: if True, abandon pre-defined fc layer
    """
    model = ReShadowNet2D(flexBottleneck, [3, 4, 6, 3], parameters, feat=feat, **kwargs)
    return model