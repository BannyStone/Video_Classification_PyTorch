import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import math

class prenormFST(nn.Module):
    def __init__(self, 
                inplanes, mid_planes, planes, 
                s_kernel_size=3, t_kernel_size=3, 
                s_stride=1, t_stride=1, 
                s_dilation=1, t_dilation=1):
        super(prenormFST, self).__init__()
        # calculate padding
        s_padding = (s_kernel_size - 1) // 2 + s_dilation - 1
        t_padding = (t_kernel_size - 1) // 2 + t_dilation - 1
        # following R(2+1)D
        self.conv_s = nn.Conv3d(inplanes, mid_planes, 
                                kernel_size=(1,) + (s_kernel_size,)*2, 
                                padding=(0,)+(s_padding,)*2, 
                                stride=(1,)+(s_stride,)*2,
                                dilation=(1,)+(s_dilation,)*2)
        self.conv_t = nn.Conv3d(mid_planes, planes, 
                                kernel_size=(t_kernel_size, 1, 1), 
                                padding=(t_padding, 0, 0), 
                                stride=(t_stride, 1, 1),
                                dilation=(t_dilation, 1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm3d(inplanes)
        self.bn2 = nn.BatchNorm3d(mid_planes)

    def forward(self, x):
        # spatial conv
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv_s(x)
        # temporal conv
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv_t(x)

        return x

class prenormFST_spt(nn.Module):
    def __init__(self, 
                inplanes, mid_planes, planes, 
                s_kernel_size=3,
                s_stride=1, t_stride=1, 
                s_dilation=1):
        super(prenormFST_spt, self).__init__()
        # calculate padding
        s_padding = (s_kernel_size - 1) // 2 + s_dilation - 1
        # following R(2+1)D
        self.conv_s = nn.Conv3d(inplanes, mid_planes, 
                                kernel_size=(1,) + (s_kernel_size,)*2, 
                                padding=(0,)+(s_padding,)*2, 
                                stride=(1,)+(s_stride,)*2,
                                dilation=(1,)+(s_dilation,)*2)
        self.conv_t = nn.Conv3d(mid_planes, planes, 
                                kernel_size=1, 
                                padding=0, 
                                stride=(t_stride, 1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm3d(inplanes)
        self.bn2 = nn.BatchNorm3d(mid_planes)

    def forward(self, x):
        # spatial conv
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv_s(x)
        # temporal conv
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv_t(x)

        return x

class sharenormFST(nn.Module):
    def __init__(self, 
                inplanes, mid_planes, planes, 
                s_kernel_size=3, t_kernel_size=3, 
                s_stride=1, t_stride=1, 
                s_dilation=1, t_dilation=1):
        super(sharenormFST, self).__init__()
        # calculate padding
        s_padding = (s_kernel_size - 1) // 2 + s_dilation - 1
        t_padding = (t_kernel_size - 1) // 2 + t_dilation - 1
        # following R(2+1)D
        self.conv_s = nn.Conv3d(inplanes, mid_planes, 
                                kernel_size=(1,) + (s_kernel_size,)*2, 
                                padding=(0,)+(s_padding,)*2, 
                                stride=(1,)+(s_stride,)*2,
                                dilation=(1,)+(s_dilation,)*2)
        self.conv_t = nn.Conv3d(mid_planes, planes, 
                                kernel_size=(t_kernel_size, 1, 1), 
                                padding=(t_padding, 0, 0), 
                                stride=(t_stride, 1, 1),
                                dilation=(t_dilation, 1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm3d(mid_planes)

    def forward(self, x):
        # spatial conv
        x = self.conv_s(x)
        # temporal conv
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv_t(x)

        return x

class FST(nn.Module):
    def __init__(self, 
                inplanes, planes, 
                s_kernel_size=3, t_kernel_size=3, 
                s_stride=1, t_stride=1, 
                s_dilation=1, t_dilation=1,
                wide=True):
        super(FST, self).__init__()
        if wide:
            c3d_params = inplanes * planes * s_kernel_size * s_kernel_size * t_kernel_size
            mid_planes = math.floor(c3d_params / (s_kernel_size * s_kernel_size * inplanes + t_kernel_size * planes))
        else:
            mid_planes = inplanes
        # reduce channel
        # if reduce_channel:
            # mid_planes = math.ceil(mid_planes / 3)
        # calculate padding
        s_padding = (s_kernel_size - 1) // 2 + s_dilation - 1
        t_padding = (t_kernel_size - 1) // 2 + t_dilation - 1
        # following R(2+1)D
        self.conv_s = nn.Conv3d(inplanes, mid_planes, 
                                kernel_size=(1,) + (s_kernel_size,)*2, 
                                padding=(0,)+(s_padding,)*2, 
                                stride=(1,)+(s_stride,)*2,
                                dilation=(1,)+(s_dilation,)*2)
        self.conv_t = nn.Conv3d(mid_planes, planes, 
                                kernel_size=(t_kernel_size, 1, 1), 
                                padding=(t_padding, 0, 0), 
                                stride=(t_stride, 1, 1),
                                dilation=(t_dilation, 1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm3d(inplanes)
        self.bn2 = nn.BatchNorm3d(mid_planes)

    def forward(self, x):
        # spatial conv
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv_s(x)
        # temporal conv
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv_t(x)

        return x