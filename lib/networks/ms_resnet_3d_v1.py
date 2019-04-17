"""
Modify the original file to make the class support feature extraction
"""
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import math
import torch.utils.model_zoo as model_zoo
from ..modules import *

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class Bottleneck3D_000(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, t_stride=1, downsample=None):
        super(Bottleneck3D_000, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, 
                               stride=[t_stride, 1, 1], bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(1, 3, 3), 
                               stride=[1, stride, stride], padding=(0, 1, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
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

class BaselineBottleneck3D_v1(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, t_stride=1, downsample=None):
        super(BaselineBottleneck3D_v1, self).__init__()
        self.conv1_t = nn.Conv3d(inplanes, planes, 
                               kernel_size=(3, 1, 1), 
                               stride=(t_stride, 1, 1),
                               padding=(2, 0, 0), 
                               bias=False,
                               dilation=(2,1,1))
        self.conv1 = nn.Conv3d(inplanes, planes, 
                               kernel_size=(3, 1, 1), 
                               stride=(t_stride, 1, 1),
                               padding=(1, 0, 0), 
                               bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, 
                               kernel_size=(1, 3, 3), 
                               stride=(1, stride, stride), 
                               padding=(0, 1, 1), 
                               bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, 
                               kernel_size=1, 
                               bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out_t = self.conv1_t(x)
        out_p = self.conv1(x)
        out = 0.5 * out_t + 0.5 * out_p
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

class BaselineBottleneck3D_v2(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, t_stride=1, downsample=None):
        super(BaselineBottleneck3D_v2, self).__init__()
        self.conv1_t1 = nn.Conv3d(inplanes, planes, 
                               kernel_size=(3, 1, 1), 
                               stride=(t_stride, 1, 1),
                               padding=(2, 0, 0), 
                               bias=False,
                               dilation=(2,1,1))
        self.conv1_t2 = nn.Conv3d(inplanes, planes, 
                               kernel_size=(3, 1, 1), 
                               stride=(t_stride, 1, 1),
                               padding=(1, 0, 0), 
                               bias=False)
        self.conv1 = nn.Conv3d(inplanes, planes, 
                               kernel_size=(1, 1, 1), 
                               stride=(t_stride, 1, 1),
                               padding=(0, 0, 0), 
                               bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, 
                               kernel_size=(1, 3, 3), 
                               stride=(1, stride, stride), 
                               padding=(0, 1, 1), 
                               bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, 
                               kernel_size=1, 
                               bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out_t1 = self.conv1_t1(x)
        out_t2 = self.conv1_t2(x)
        out_p = self.conv1(x)
        out = out_t1/3 + out_t2/3 + out_p/3
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

class AdaResNet3D(nn.Module):

    def __init__(self, block, layers, num_classes=1000, feat=False, **kwargs):
        if not isinstance(block, list):
            block = [block] * 4
        else:
            assert(len(block)) == 4, "Block number must be 4 for ResNet-Stype networks."
        self.inplanes = 64
        super(AdaResNet3D, self).__init__()
        self.feat = feat
        self.conv1 = nn.Conv3d(3, 64, 
                               kernel_size=(1, 7, 7), 
                               stride=(1, 2, 2), 
                               padding=(0, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), 
                                    stride=(1, 2, 2), 
                                    padding=(0, 1, 1))
        self.layer1 = self._make_layer(block[0], 64, layers[0])
        self.layer2 = self._make_layer(block[1], 128, layers[1], stride=2, t_stride=2)
        self.layer3 = self._make_layer(block[2], 256, layers[2], stride=2, t_stride=2)
        self.layer4 = self._make_layer(block[3], 512, layers[3], stride=2, t_stride=2)
        self.avgpool = GloAvgPool3d()
        self.feat_dim = 512 * block[0].expansion
        if not feat:
            self.fc = nn.Linear(512 * block[0].expansion, num_classes)

        for n, m in self.named_modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d) and "conv_t" not in n:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, Scale3d):
                nn.init.constant_(m.scale, 0)


    def _make_layer(self, block, planes, blocks, stride=1, t_stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=(t_stride, stride, stride), bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, t_stride=t_stride, downsample=downsample))
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


def part_state_dict(state_dict, model_dict):
    added_dict = {}
    for k, v in state_dict.items():
        if ".conv1.weight" in k:
            new_k = k[:k.index(".conv1.weight")]+'.conv1_t.weight'
            added_dict.update({new_k: v})
            new_k = k[:k.index(".conv1.weight")]+'.conv1_t1.weight'
            added_dict.update({new_k: v})
            new_k = k[:k.index(".conv1.weight")]+'.conv1_t2.weight'
            added_dict.update({new_k: v})
    state_dict.update(added_dict)
    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    pretrained_dict = inflate_state_dict(pretrained_dict, model_dict)
    model_dict.update(pretrained_dict)
    return model_dict


def inflate_state_dict(pretrained_dict, model_dict):
    for k in pretrained_dict.keys():
        if pretrained_dict[k].size() != model_dict[k].size():
            assert(pretrained_dict[k].size()[:2] == model_dict[k].size()[:2]), \
                   "To inflate, channel number should match."
            assert(pretrained_dict[k].size()[-2:] == model_dict[k].size()[-2:]), \
                   "To inflate, spatial kernel size should match."
            print("Layer {} needs inflation.".format(k))
            shape = list(pretrained_dict[k].shape)
            shape.insert(2, 1)
            t_length = model_dict[k].shape[2]
            pretrained_dict[k] = pretrained_dict[k].reshape(shape)
            if t_length != 1:
                pretrained_dict[k] = pretrained_dict[k].expand_as(model_dict[k]) / t_length
            assert(pretrained_dict[k].size() == model_dict[k].size()), \
                   "After inflation, model shape should match."

    return pretrained_dict

def ms_resnet26_3d_v1(pretrained=False, feat=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AdaResNet3D([BaselineBottleneck3D_v1, BaselineBottleneck3D_v1, BaselineBottleneck3D_v1, BaselineBottleneck3D_v1], 
                     [2, 2, 2, 2], feat=feat, **kwargs)
    if pretrained:
        if kwargs['pretrained_model'] is None:
            pass
            # state_dict = model_zoo.load_url(model_urls['resnet50'])
        else:
            print("Using specified pretrain model")
            state_dict = kwargs['pretrained_model']
        if feat:
            new_state_dict = part_state_dict(state_dict, model.state_dict())
            model.load_state_dict(new_state_dict)
    return model

def ms_resnet26_3d_v2(pretrained=False, feat=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AdaResNet3D([BaselineBottleneck3D_v2, BaselineBottleneck3D_v2, BaselineBottleneck3D_v2, BaselineBottleneck3D_v2], 
                     [2, 2, 2, 2], feat=feat, **kwargs)
    if pretrained:
        if kwargs['pretrained_model'] is None:
            pass
            # state_dict = model_zoo.load_url(model_urls['resnet50'])
        else:
            print("Using specified pretrain model")
            state_dict = kwargs['pretrained_model']
        if feat:
            new_state_dict = part_state_dict(state_dict, model.state_dict())
            model.load_state_dict(new_state_dict)
    return model