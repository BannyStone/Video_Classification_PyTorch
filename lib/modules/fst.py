import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.model_zoo as model_zoo
import math

class GloSptMaxPool3d(nn.Module):
    def __init__(self):
        super(GloSptMaxPool3d, self).__init__()
        self.stride = 1
        self.padding = 0
        self.ceil_mode = False
        self.count_include_pad = True

    def forward(self, input):
        input_shape = input.shape
        kernel_size = (1,) + input_shape[3:]
        return F.max_pool3d(input, kernel_size=kernel_size, stride=self.stride,
                            padding=self.padding, ceil_mode=self.ceil_mode)

class Rectification(nn.Module):
    def __init__(self):
        super(Rectification, self).__init__()

    def forward(self, l, g):
        b, c, t, _, _ = l.size()
        g = g.view(b, c, t, 1, 1)
        return l * g.expand_as(l)

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

class sharenormFST_spt(nn.Module):
    def __init__(self, 
                inplanes, mid_planes, planes, 
                s_kernel_size=3, 
                s_stride=1, t_stride=1, 
                s_dilation=1):
        super(sharenormFST_spt, self).__init__()
        # calculate padding
        s_padding = (s_kernel_size - 1) // 2 + s_dilation - 1
        # following R(2+1)D
        self.conv_s = nn.Conv3d(inplanes, mid_planes, 
                                kernel_size=(1,) + (s_kernel_size,)*2, 
                                padding=(0,)+(s_padding,)*2, 
                                stride=(1,)+(s_stride,)*2,
                                dilation=(1,)+(s_dilation,)*2)
        self.conv_p = nn.Conv3d(mid_planes, planes, 
                                kernel_size=1, 
                                padding=0, 
                                stride=(t_stride, 1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm3d(mid_planes)

    def forward(self, x):
        # spatial conv
        x = self.conv_s(x)
        # temporal conv
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv_p(x)

        return x

class sharenormGSV_spt(nn.Module):
    def __init__(self, 
                inplanes, mid_planes, planes, 
                s_kernel_size=3, t_kernel_size=3,
                s_stride=1, t_stride=1, 
                s_dilation=1, t_dilation=1):
        super(sharenormGSV_spt, self).__init__()
        # calculate padding
        s_padding = (s_kernel_size - 1) // 2 + s_dilation - 1
        t_padding = (t_kernel_size - 1) // 2 + t_dilation - 1
        # following R(2+1)D
        self.fst_spt = sharenormFST_spt(inplanes, mid_planes, planes, 
                                        s_kernel_size=s_kernel_size,
                                        s_stride=s_stride, t_stride=t_stride,
                                        s_dilation=s_dilation)
        self.conv_t = nn.Conv3d(inplanes, planes,
                                kernel_size=(t_kernel_size, 1, 1),
                                padding=(t_padding, 0, 0),
                                stride=(t_stride, 1, 1),
                                dilation=(t_dilation, 1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm3d(inplanes)
        self.spt_sum = GloSptMaxPool3d()
        self.rect = Rectification()

    def forward(self, x):
        # sharenorm
        x = self.bn(x)
        x = self.relu(x)
        # lsv branch
        l = self.fst_spt(x)
        # gsv branch
        g = self.spt_sum(x)
        g = self.conv_t(g)
        g = self.sigmoid(g)
        # rectification
        l = self.rect(l, g)

        return l

class sharenormGSV(nn.Module):
    def __init__(self, 
                inplanes, mid_planes, planes, 
                s_kernel_size=3, t_kernel_size=3,
                s_stride=1, t_stride=1, 
                s_dilation=1, t_dilation=1):
        super(sharenormGSV, self).__init__()
        # calculate padding
        s_padding = (s_kernel_size - 1) // 2 + s_dilation - 1
        t_padding = (t_kernel_size - 1) // 2 + t_dilation - 1
        # following R(2+1)D
        self.fst_spt = sharenormFST(inplanes, mid_planes, planes, 
                                    s_kernel_size=s_kernel_size, t_kernel_size=t_kernel_size,
                                    s_stride=s_stride, t_stride=t_stride,
                                    s_dilation=s_dilation, t_dilation=t_dilation)
        self.conv_t = nn.Conv3d(inplanes, planes,
                                kernel_size=(t_kernel_size, 1, 1),
                                padding=(t_padding, 0, 0),
                                stride=(t_stride, 1, 1),
                                dilation=(t_dilation, 1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm3d(inplanes)
        self.spt_sum = GloSptMaxPool3d()
        self.rect = Rectification()

    def forward(self, x):
        # sharenorm
        x = self.bn(x)
        x = self.relu(x)
        # lsv branch
        l = self.fst_spt(x)
        # gsv branch
        g = self.spt_sum(x)
        g = self.conv_t(g)
        g = self.sigmoid(g)
        # rectification
        l = self.rect(l, g)

        return l

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