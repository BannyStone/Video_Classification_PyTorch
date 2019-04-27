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

class AdaModule_v1(nn.Module):
    """
    Compress spatial dimension to learn temporal dependency.
    Switch between 3x1x1 and 1x1x1
    """
    def __init__(self, inplanes, planes, t_stride):
        super(AdaModule_v1, self).__init__()
        self.spt_sum = GloSptMaxPool3d()
        self.conv_t1 = nn.Conv3d(inplanes, 2 * planes, 
                                kernel_size=(3,1,1),
                                stride=(t_stride,1,1), 
                                padding=(1,0,0), 
                                bias=False)
        self.bn_t1 = nn.BatchNorm3d(2 * planes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        x = self.spt_sum(input)
        x = self.conv_t1(x)
        # x = self.bn_t1(x)# (N, 2C, T, 1, 1)
        N, C, T, _, __ = x.shape
        C //= 2
        x = x.view(N, 2, -1) # (N, 2, CT)
        x = self.softmax(x)
        out_shape = (N, C, T, 1, 1)
        out1 = x[:,0,:].view(out_shape)
        out2 = x[:,1,:].view(out_shape)

        return out1, out2

class AdaModule_v1_1(nn.Module):
    """
    Compress spatial dimension to learn temporal dependency.
    Switch between 3x1x1 and 1x1x1
    """
    def __init__(self, inplanes, planes, t_stride, factor=4):
        super(AdaModule_v1_1, self).__init__()
        self.spt_sum = GloSptMaxPool3d()
        self.conv_p1 = nn.Conv3d(inplanes, planes//factor,
                                kernel_size=(1,1,1),
                                stride=(1,1,1),
                                padding=(0,0,0),
                                bias=False)
        self.bn_p1 = nn.BatchNorm3d(planes//factor)
        self.conv_t1 = nn.Conv3d(planes//factor, 2 * planes, 
                                kernel_size=(3,1,1),
                                stride=(t_stride,1,1), 
                                padding=(1,0,0), 
                                bias=False)
        self.relu = nn.ReLU(inplace=True)
        # self.bn_t1 = nn.BatchNorm3d(2 * planes)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        x = self.spt_sum(input)
        x = self.conv_p1(x)
        x = self.bn_p1(x)
        x = self.relu(x)
        x = self.conv_t1(x)
        # x = self.bn_t1(x)# (N, 2C, T, 1, 1)
        N, C, T, _, __ = x.shape
        C //= 2
        x = x.view(N, 2, -1) # (N, 2, CT)
        x = self.softmax(x)
        out_shape = (N, C, T, 1, 1)
        out1 = x[:,0,:].view(out_shape)
        out2 = x[:,1,:].view(out_shape)

        return out1, out2

class AdaModule_v1_1_1(nn.Module):
    """
    Compress spatial dimension to learn temporal dependency.
    Switch between 3x1x1 and 1x1x1
    """
    def __init__(self, inplanes, planes, t_stride, factor=4):
        super(AdaModule_v1_1_1, self).__init__()
        self.spt_sum = GloSptMaxPool3d()
        self.conv_p1 = nn.Conv3d(inplanes, planes//factor,
                                kernel_size=(1,1,1),
                                stride=(1,1,1),
                                padding=(0,0,0),
                                bias=False)
        self.bn_p1 = nn.BatchNorm3d(planes//factor)
        self.conv_t1 = nn.Conv3d(planes//factor, planes//factor, 
                                kernel_size=(3,1,1),
                                stride=(t_stride,1,1), 
                                padding=(1,0,0), 
                                bias=False)
        self.bn_t1 = nn.BatchNorm3d(planes//factor)
        self.conv_p2 = nn.Conv3d(planes//factor, 2*planes,
                                kernel_size=(1,1,1),
                                stride=(1,1,1),
                                padding=(0,0,0),
                                bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        x = self.spt_sum(input)
        # p1
        x = self.conv_p1(x)
        x = self.bn_p1(x)
        x = self.relu(x)
        # t1
        x = self.conv_t1(x)
        x = self.bn_t1(x)
        x = self.relu(x)
        # p2
        x = self.conv_p2(x)
        # x = self.bn_t1(x)# (N, 2C, T, 1, 1)
        N, C, T, _, __ = x.shape
        C //= 2
        x = x.view(N, 2, -1) # (N, 2, CT)
        x = self.softmax(x)
        out_shape = (N, C, T, 1, 1)
        out1 = x[:,0,:].view(out_shape)
        out2 = x[:,1,:].view(out_shape)

        return out1, out2

class AdaModule_v1_2(nn.Module):
    """
    Compress spatial dimension to learn temporal dependency.
    Switch between 3x1x1 and 1x1x1
    free dimension: convs
    """
    def __init__(self, inplanes, planes, t_stride, factor=4):
        super(AdaModule_v1_2, self).__init__()
        self.spt_sum = GloSptMaxPool3d()
        self.tem_sum = GloAvgPool3d()
        self.conv_p1 = nn.Conv3d(inplanes, planes//factor,
                                kernel_size=(1,1,1),
                                stride=(1,1,1),
                                padding=(0,0,0),
                                bias=False)
        self.bn_p1 = nn.BatchNorm3d(planes//factor)
        self.conv_t1 = nn.Conv3d(planes//factor, planes//factor, 
                                kernel_size=(3,1,1),
                                stride=(t_stride,1,1), 
                                padding=(1,0,0), 
                                bias=False)
        self.bn_t1 = nn.BatchNorm3d(planes//factor)
        self.conv_p2 = nn.Conv3d(planes//factor, 2,
                                kernel_size=(1,1,1),
                                stride=(1,1,1),
                                padding=(0,0,0),
                                bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        x = self.spt_sum(input)
        # p1
        x = self.conv_p1(x)
        x = self.bn_p1(x)
        x = self.relu(x)
        # t1
        x = self.conv_t1(x)
        x = self.bn_t1(x)
        x = self.relu(x)
        # global pooling
        x = self.tem_sum(x)
        # p2
        x = self.conv_p2(x)

        N = x.shape[0]
        x = x.view(N, 2) # (N, 2)
        x = self.softmax(x)
        out_shape = (N, 1, 1, 1, 1)
        out1 = x[:,0].view(out_shape)
        out2 = x[:,1].view(out_shape)

        return out1, out2

class AdaModule_v1_3(nn.Module):
    """
    Compress spatial dimension to learn temporal dependency.
    Switch between 3x1x1 and 1x1x1
    free dimension: temporal
    """
    def __init__(self, inplanes, planes, t_stride, factor=4):
        super(AdaModule_v1_3, self).__init__()
        self.spt_sum = GloSptMaxPool3d()
        # self.tem_sum = GloAvgPool3d()
        self.conv_p1 = nn.Conv3d(inplanes, planes//factor,
                                kernel_size=(1,1,1),
                                stride=(1,1,1),
                                padding=(0,0,0),
                                bias=False)
        self.bn_p1 = nn.BatchNorm3d(planes//factor)
        self.conv_t1 = nn.Conv3d(planes//factor, planes//factor, 
                                kernel_size=(3,1,1),
                                stride=(t_stride,1,1), 
                                padding=(1,0,0), 
                                bias=False)
        self.bn_t1 = nn.BatchNorm3d(planes//factor)
        self.conv_p2 = nn.Conv3d(planes//factor, 2,
                                kernel_size=(1,1,1),
                                stride=(1,1,1),
                                padding=(0,0,0),
                                bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        x = self.spt_sum(input)
        # p1
        x = self.conv_p1(x)
        x = self.bn_p1(x)
        x = self.relu(x)
        # t1
        x = self.conv_t1(x)
        x = self.bn_t1(x)
        x = self.relu(x)
        # p2
        x = self.conv_p2(x)

        N, _, T = x.shape[:3]
        x = x.view(N, 2, -1) # (N, 2, T)
        x = self.softmax(x)
        out_shape = (N, 1, T, 1, 1)
        out1 = x[:,0,:].view(out_shape)
        out2 = x[:,1,:].view(out_shape)

        return out1, out2

class AdaModule_v1_4(nn.Module):
    """
    Compress spatial dimension to learn temporal dependency.
    Switch between 3x1x1 and 1x1x1
    free dimension: channel
    """
    def __init__(self, inplanes, planes, t_stride, factor=4):
        super(AdaModule_v1_4, self).__init__()
        self.spt_sum = GloSptMaxPool3d()
        self.tem_sum = GloAvgPool3d()
        self.conv_p1 = nn.Conv3d(inplanes, planes//factor,
                                kernel_size=(1,1,1),
                                stride=(1,1,1),
                                padding=(0,0,0),
                                bias=False)
        self.bn_p1 = nn.BatchNorm3d(planes//factor)
        self.conv_t1 = nn.Conv3d(planes//factor, planes//factor, 
                                kernel_size=(3,1,1),
                                stride=(t_stride,1,1), 
                                padding=(1,0,0), 
                                bias=False)
        self.bn_t1 = nn.BatchNorm3d(planes//factor)
        self.conv_p2 = nn.Conv3d(planes//factor, 2 * planes,
                                kernel_size=(1,1,1),
                                stride=(1,1,1),
                                padding=(0,0,0),
                                bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        x = self.spt_sum(input)
        # p1
        x = self.conv_p1(x)
        x = self.bn_p1(x)
        x = self.relu(x)
        # t1
        x = self.conv_t1(x)
        x = self.bn_t1(x)
        x = self.relu(x)
        # global pooling
        x = self.tem_sum(x)
        # p2
        x = self.conv_p2(x)

        N, C = x.shape[:2]
        C //= 2
        x = x.view(N, 2, C) # (N, 2, C)
        x = self.softmax(x)
        out_shape = (N, C, 1, 1, 1)
        out1 = x[:,0,:].view(out_shape)
        out2 = x[:,1,:].view(out_shape)

        return out1, out2

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

class BaselineBottleneck3D(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, t_stride=1, downsample=None):
        super(BaselineBottleneck3D, self).__init__()
        self.conv1_t = nn.Conv3d(inplanes, planes, 
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

class AdaBottleneck3D(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, t_stride=1, downsample=None):
        super(AdaBottleneck3D, self).__init__()
        self.ada_m = AdaModule_v1(inplanes, planes, t_stride=t_stride)
        self.conv1_t = nn.Conv3d(inplanes, planes, 
                               kernel_size=(3, 1, 1), 
                               stride=(t_stride, 1, 1),
                               padding=(1, 0, 0), 
                               bias=False)
        self.bn1_t = nn.BatchNorm3d(planes)
        self.conv1 = nn.Conv3d(inplanes, planes, 
                               kernel_size=(1, 1, 1), 
                               stride=(t_stride, 1, 1),
                               padding=(0, 0, 0), 
                               bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        # self.bn1_t = nn.BatchNorm3d(planes)
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
        self.spt_glo_pool = GloSptMaxPool3d()
        self.conv_t = nn.Conv3d(inplanes, planes, 
                                kernel_size=(3,1,1),
                                stride=(t_stride,1,1),
                                padding=(1,0,0),
                                bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out_p = self.conv1(x)
        out_p = self.bn1(out_p)
        out_p = self.relu(out_p)
        out_t = self.conv1_t(x)
        out_t = self.bn1_t(out_t)
        out_t = self.relu(out_t)

        guid1, guid2 = self.ada_m(x)
        out = guid1 * out_p + guid2 * out_t

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

class AdaBottleneck3D_v1_1(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, t_stride=1, downsample=None):
        super(AdaBottleneck3D_v1_1, self).__init__()
        self.ada_m = AdaModule_v1_1(inplanes, planes, t_stride=t_stride, factor=4)
        self.conv1_t = nn.Conv3d(inplanes, planes, 
                               kernel_size=(3, 1, 1), 
                               stride=(t_stride, 1, 1),
                               padding=(1, 0, 0), 
                               bias=False)
        self.bn1_t = nn.BatchNorm3d(planes)
        self.conv1 = nn.Conv3d(inplanes, planes, 
                               kernel_size=(1, 1, 1), 
                               stride=(t_stride, 1, 1),
                               padding=(0, 0, 0), 
                               bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        # self.bn1_t = nn.BatchNorm3d(planes)
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
        self.spt_glo_pool = GloSptMaxPool3d()
        self.conv_t = nn.Conv3d(inplanes, planes, 
                                kernel_size=(3,1,1),
                                stride=(t_stride,1,1),
                                padding=(1,0,0),
                                bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out_p = self.conv1(x)
        out_p = self.bn1(out_p)
        out_p = self.relu(out_p)
        out_t = self.conv1_t(x)
        out_t = self.bn1_t(out_t)
        out_t = self.relu(out_t)

        guid1, guid2 = self.ada_m(x)
        out = guid1 * out_p + guid2 * out_t

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

class AdaBottleneck3D_v1_1_1(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, t_stride=1, downsample=None):
        super(AdaBottleneck3D_v1_1_1, self).__init__()
        self.ada_m = AdaModule_v1_1_1(inplanes, planes, t_stride=t_stride, factor=4)
        self.conv1_t = nn.Conv3d(inplanes, planes, 
                               kernel_size=(3, 1, 1), 
                               stride=(t_stride, 1, 1),
                               padding=(1, 0, 0), 
                               bias=False)
        self.bn1_t = nn.BatchNorm3d(planes)
        self.conv1 = nn.Conv3d(inplanes, planes, 
                               kernel_size=(1, 1, 1), 
                               stride=(t_stride, 1, 1),
                               padding=(0, 0, 0), 
                               bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        # self.bn1_t = nn.BatchNorm3d(planes)
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
        self.spt_glo_pool = GloSptMaxPool3d()
        self.conv_t = nn.Conv3d(inplanes, planes, 
                                kernel_size=(3,1,1),
                                stride=(t_stride,1,1),
                                padding=(1,0,0),
                                bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out_p = self.conv1(x)
        out_p = self.bn1(out_p)
        out_p = self.relu(out_p)
        out_t = self.conv1_t(x)
        out_t = self.bn1_t(out_t)
        out_t = self.relu(out_t)

        guid1, guid2 = self.ada_m(x)
        out = guid1 * out_p + guid2 * out_t

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

class AdaBottleneck3D_TC_v1(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, t_stride=1, downsample=None):
        super(AdaBottleneck3D_TC_v1, self).__init__()
        self.ada_m = AdaModule_v1_1_1(inplanes, planes, t_stride=t_stride, factor=4)
        self.conv1_t = nn.Conv3d(inplanes, planes, 
                               kernel_size=(3, 1, 1), 
                               stride=(t_stride, 1, 1),
                               padding=(1, 0, 0), 
                               bias=False)
        self.bn1_t = nn.BatchNorm3d(planes)
        self.conv1 = nn.Conv3d(inplanes, planes, 
                               kernel_size=(1, 1, 1), 
                               stride=(t_stride, 1, 1),
                               padding=(0, 0, 0), 
                               bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        # self.bn1_t = nn.BatchNorm3d(planes)
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
        self.spt_glo_pool = GloSptMaxPool3d()
        self.conv_t = nn.Conv3d(inplanes, planes, 
                                kernel_size=(3,1,1),
                                stride=(t_stride,1,1),
                                padding=(1,0,0),
                                bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out_p = self.conv1(x)
        out_p = self.bn1(out_p)
        out_p = self.relu(out_p)
        out_t = self.conv1_t(x)
        out_t = self.bn1_t(out_t)
        out_t = self.relu(out_t)

        guid1, guid2 = self.ada_m(x)
        out = guid1 * out_p + guid2 * out_t

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

class AdaBottleneck3D_v1_2(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, t_stride=1, downsample=None):
        super(AdaBottleneck3D_v1_2, self).__init__()
        self.ada_m = AdaModule_v1_2(inplanes, planes, t_stride=t_stride, factor=4)
        self.conv1_t = nn.Conv3d(inplanes, planes, 
                               kernel_size=(3, 1, 1), 
                               stride=(t_stride, 1, 1),
                               padding=(1, 0, 0), 
                               bias=False)
        self.bn1_t = nn.BatchNorm3d(planes)
        self.conv1 = nn.Conv3d(inplanes, planes, 
                               kernel_size=(1, 1, 1), 
                               stride=(t_stride, 1, 1),
                               padding=(0, 0, 0), 
                               bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        # self.bn1_t = nn.BatchNorm3d(planes)
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
        self.spt_glo_pool = GloSptMaxPool3d()
        self.conv_t = nn.Conv3d(inplanes, planes, 
                                kernel_size=(3,1,1),
                                stride=(t_stride,1,1),
                                padding=(1,0,0),
                                bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out_p = self.conv1(x)
        out_p = self.bn1(out_p)
        out_p = self.relu(out_p)
        out_t = self.conv1_t(x)
        out_t = self.bn1_t(out_t)
        out_t = self.relu(out_t)

        guid1, guid2 = self.ada_m(x)
        out = guid1 * out_p + guid2 * out_t

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

class AdaBottleneck3D_v1_3(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, t_stride=1, downsample=None):
        super(AdaBottleneck3D_v1_3, self).__init__()
        self.ada_m = AdaModule_v1_3(inplanes, planes, t_stride=t_stride, factor=4)
        self.conv1_t = nn.Conv3d(inplanes, planes, 
                               kernel_size=(3, 1, 1), 
                               stride=(t_stride, 1, 1),
                               padding=(1, 0, 0), 
                               bias=False)
        self.bn1_t = nn.BatchNorm3d(planes)
        self.conv1 = nn.Conv3d(inplanes, planes, 
                               kernel_size=(1, 1, 1), 
                               stride=(t_stride, 1, 1),
                               padding=(0, 0, 0), 
                               bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        # self.bn1_t = nn.BatchNorm3d(planes)
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
        self.spt_glo_pool = GloSptMaxPool3d()
        self.conv_t = nn.Conv3d(inplanes, planes, 
                                kernel_size=(3,1,1),
                                stride=(t_stride,1,1),
                                padding=(1,0,0),
                                bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out_p = self.conv1(x)
        out_p = self.bn1(out_p)
        out_p = self.relu(out_p)
        out_t = self.conv1_t(x)
        out_t = self.bn1_t(out_t)
        out_t = self.relu(out_t)

        guid1, guid2 = self.ada_m(x)
        out = guid1 * out_p + guid2 * out_t

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

class AdaBottleneck3D_v1_4(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, t_stride=1, downsample=None):
        super(AdaBottleneck3D_v1_4, self).__init__()
        self.ada_m = AdaModule_v1_4(inplanes, planes, t_stride=t_stride, factor=4)
        self.conv1_t = nn.Conv3d(inplanes, planes, 
                               kernel_size=(3, 1, 1), 
                               stride=(t_stride, 1, 1),
                               padding=(1, 0, 0), 
                               bias=False)
        self.bn1_t = nn.BatchNorm3d(planes)
        self.conv1 = nn.Conv3d(inplanes, planes, 
                               kernel_size=(1, 1, 1), 
                               stride=(t_stride, 1, 1),
                               padding=(0, 0, 0), 
                               bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        # self.bn1_t = nn.BatchNorm3d(planes)
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
        self.spt_glo_pool = GloSptMaxPool3d()
        self.conv_t = nn.Conv3d(inplanes, planes, 
                                kernel_size=(3,1,1),
                                stride=(t_stride,1,1),
                                padding=(1,0,0),
                                bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out_p = self.conv1(x)
        out_p = self.bn1(out_p)
        out_p = self.relu(out_p)
        out_t = self.conv1_t(x)
        out_t = self.bn1_t(out_t)
        out_t = self.relu(out_t)

        guid1, guid2 = self.ada_m(x)
        out = guid1 * out_p + guid2 * out_t

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
        if ".conv1." in k:
            if ".conv1.weight" in k:
                new_k = k[:k.index(".conv1.weight")]+'.conv1_t.weight'
                added_dict.update({new_k: v})
            elif ".conv1.bias" in k:
                new_k = k[:k.index(".conv1.bias")]+'.conv1_t.bias'
                added_dict.update({new_k: v})
            else:
                raise ValueError("Invalid param or buffer for Conv Layer")
        elif ".bn1." in k:
            if ".bn1.weight" in k:
                new_k = k[:k.index(".bn1.weight")]+'.bn1_t.weight'
                added_dict.update({new_k: v})
            elif ".bn1.bias" in k:
                new_k = k[:k.index(".bn1.bias")]+'.bn1_t.bias'
                added_dict.update({new_k: v})
            elif ".bn1.running_mean" in k:
                new_k = k[:k.index(".bn1.running_mean")]+'.bn1_t.running_mean'
                added_dict.update({new_k: v})
            elif ".bn1.running_var" in k:
                new_k = k[:k.index(".bn1.running_var")]+'.bn1_t.running_var'
                added_dict.update({new_k: v})
            elif ".bn1.num_batches_tracked" in k:
                new_k = k[:k.index(".bn1.num_batches_tracked")]+'.bn1_t.num_batches_tracked'
                added_dict.update({new_k: v})
            else:
                raise ValueError("Invalid param or buffer for BN Layer")


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

def ada_resnet26_3d_v1(pretrained=False, feat=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AdaResNet3D([AdaBottleneck3D, AdaBottleneck3D, AdaBottleneck3D, AdaBottleneck3D], 
                     [2, 2, 2, 2], feat=feat, **kwargs)
    if pretrained:
        if kwargs['pretrained_model'] is None:
            raise ValueError("For resnet26, pretrained model must be specified.")
            # state_dict = model_zoo.load_url(model_urls['resnet50'])
        else:
            print("Using specified pretrain model")
            state_dict = kwargs['pretrained_model']
        if feat:
            new_state_dict = part_state_dict(state_dict, model.state_dict())
            model.load_state_dict(new_state_dict)
    return model

def ada_resnet26_3d_v1_1(pretrained=False, feat=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AdaResNet3D([AdaBottleneck3D_v1_1, AdaBottleneck3D_v1_1, AdaBottleneck3D_v1_1, AdaBottleneck3D_v1_1], 
                     [2, 2, 2, 2], feat=feat, **kwargs)
    if pretrained:
        if kwargs['pretrained_model'] is None:
            raise ValueError("For resnet26, pretrained model must be specified.")
            # state_dict = model_zoo.load_url(model_urls['resnet50'])
        else:
            print("Using specified pretrain model")
            state_dict = kwargs['pretrained_model']
        if feat:
            new_state_dict = part_state_dict(state_dict, model.state_dict())
            model.load_state_dict(new_state_dict)
    return model

def ada_resnet26_3d_v1_1_1(pretrained=False, feat=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AdaResNet3D([AdaBottleneck3D_v1_1_1, AdaBottleneck3D_v1_1_1, AdaBottleneck3D_v1_1_1, AdaBottleneck3D_v1_1_1], 
                     [2, 2, 2, 2], feat=feat, **kwargs)
    if pretrained:
        if kwargs['pretrained_model'] is None:
            raise ValueError("For resnet26, pretrained model must be specified.")
            # state_dict = model_zoo.load_url(model_urls['resnet50'])
        else:
            print("Using specified pretrain model")
            state_dict = kwargs['pretrained_model']
        if feat:
            new_state_dict = part_state_dict(state_dict, model.state_dict())
            model.load_state_dict(new_state_dict)
    return model

def ada_resnet50_3d_v1_1_1(pretrained=False, feat=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AdaResNet3D([AdaBottleneck3D_v1_1_1, AdaBottleneck3D_v1_1_1, AdaBottleneck3D_v1_1_1, AdaBottleneck3D_v1_1_1], 
                     [3, 4, 6, 3], feat=feat, **kwargs)
    if pretrained:
        if kwargs['pretrained_model'] is None:
            # raise ValueError("For resnet26, pretrained model must be specified.")
            state_dict = model_zoo.load_url(model_urls['resnet50'])
        else:
            print("Using specified pretrain model")
            state_dict = kwargs['pretrained_model']
        if feat:
            new_state_dict = part_state_dict(state_dict, model.state_dict())
            model.load_state_dict(new_state_dict)
    return model

def ada_resnet26_3d_v1_2(pretrained=False, feat=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AdaResNet3D([AdaBottleneck3D_v1_2, AdaBottleneck3D_v1_2, AdaBottleneck3D_v1_2, AdaBottleneck3D_v1_2], 
                     [2, 2, 2, 2], feat=feat, **kwargs)
    if pretrained:
        if kwargs['pretrained_model'] is None:
            raise ValueError("For resnet26, pretrained model must be specified.")
            # state_dict = model_zoo.load_url(model_urls['resnet50'])
        else:
            print("Using specified pretrain model")
            state_dict = kwargs['pretrained_model']
        if feat:
            new_state_dict = part_state_dict(state_dict, model.state_dict())
            model.load_state_dict(new_state_dict)
    return model

def ada_resnet26_3d_v1_3(pretrained=False, feat=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AdaResNet3D([AdaBottleneck3D_v1_3, AdaBottleneck3D_v1_3, AdaBottleneck3D_v1_3, AdaBottleneck3D_v1_3], 
                     [2, 2, 2, 2], feat=feat, **kwargs)
    if pretrained:
        if kwargs['pretrained_model'] is None:
            raise ValueError("For resnet26, pretrained model must be specified.")
            # state_dict = model_zoo.load_url(model_urls['resnet50'])
        else:
            print("Using specified pretrain model")
            state_dict = kwargs['pretrained_model']
        if feat:
            new_state_dict = part_state_dict(state_dict, model.state_dict())
            model.load_state_dict(new_state_dict)
    return model

def ada_resnet26_3d_v1_4(pretrained=False, feat=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AdaResNet3D([AdaBottleneck3D_v1_4, AdaBottleneck3D_v1_4, AdaBottleneck3D_v1_4, AdaBottleneck3D_v1_4], 
                     [2, 2, 2, 2], feat=feat, **kwargs)
    if pretrained:
        if kwargs['pretrained_model'] is None:
            raise ValueError("For resnet26, pretrained model must be specified.")
            # state_dict = model_zoo.load_url(model_urls['resnet50'])
        else:
            print("Using specified pretrain model")
            state_dict = kwargs['pretrained_model']
        if feat:
            new_state_dict = part_state_dict(state_dict, model.state_dict())
            model.load_state_dict(new_state_dict)
    return model