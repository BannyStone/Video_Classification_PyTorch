"""
Modify the original file to make the class support feature extraction
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.model_zoo as model_zoo

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class GloAvgPool2d(nn.Module):
    def __init__(self):
        super(GloAvgPool2d, self).__init__()
        self.stride = 1
        self.padding = 0
        self.ceil_mode = False
        self.count_include_pad = True

    def forward(self, input):
        input_shape = input.shape
        kernel_size = input_shape[2:]
        return F.avg_pool2d(input, kernel_size, self.stride,
                            self.padding, self.ceil_mode, self.count_include_pad)


class GloMaxPool2d(nn.Module):
    def __init__(self):
        super(GloMaxPool2d, self).__init__()
        self.stride = 1
        self.padding = 0
        self.ceil_mode = False
        self.count_include_pad = True

    def forward(self, input):
        input_shape = input.shape
        kernel_size = input_shape[2:]
        return F.max_pool2d(input, kernel_size=kernel_size, stride=self.stride,
                            padding=self.padding, ceil_mode=self.ceil_mode)


class GSVBottleneck2D(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(GSVBottleneck2D, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes,
                               kernel_size=1,
                               stride=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion,
                               kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.spt_glo_pool = GloMaxPool2d()
        self.conv_t = nn.Conv2d(inplanes, planes, 
                                kernel_size=1,
                                stride=1,
                                bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)

        gsv = self.spt_glo_pool(x)
        gsv = self.conv_t(gsv)
        gsv = self.sigmoid(gsv)
        out = out * gsv + out

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

class GSV_ResNet2D(nn.Module):

    def __init__(self, block, layers, feat=False, **kwargs):
        if not isinstance(block, list):
            block = [block] * 4
        else:
            assert(len(block)) == 4, "Block number must be 4 for ResNet-Stype networks."
        self.inplanes = 64
        super(GSV_ResNet2D, self).__init__()
        self.feat = feat
        self.conv1 = nn.Conv2d(3, 64,
                               kernel_size=7,
                               stride=2,
                               padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3,
                                    stride=2,
                                    padding=1)
        self.layer1 = self._make_layer(block[0], 64, layers[0])
        self.layer2 = self._make_layer(block[1], 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block[2], 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block[3], 512, layers[3], stride=2)
        self.avgpool = GloAvgPool2d()
        self.feat_dim = 512 * block[0].expansion
        if not feat:
            self.fc = nn.Linear(512 * block[0].expansion, kwargs['num_classes'])

        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample))
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

def gsv_resnet50_2d_v3(pretrained=False, pretrain_model=None, feat=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = GSV_ResNet2D([GSVBottleneck2D]*4, 
                     [3, 4, 6, 3], feat=feat, **kwargs)
    if pretrained:
        import pdb
        pdb.set_trace()
        model.load_state_dict(torch.load(pretrain_model))
    return model
