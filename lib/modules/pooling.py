import torch
import torch.nn as nn
import torch.nn.functional as F

class GloAvgPool3d(nn.Module):
    def __init__(self):
        super(GloAvgPool3d, self).__init__()
        self.stride = 1
        self.padding = 0
        self.ceil_mode = False
        self.count_include_pad = True

    def forward(self, input):
        input_shape = input.shape
        kernel_size = input_shape[2:]
        return F.avg_pool3d(input, kernel_size, self.stride,
                            self.padding, self.ceil_mode, self.count_include_pad)

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

class GloSptAvgPool3d(nn.Module):
    def __init__(self):
        super(GloSptAvgPool3d, self).__init__()
        self.stride = 1
        self.padding = 0
        self.ceil_mode = False
        self.count_include_pad = True

    def forward(self, input):
        input_shape = input.shape
        kernel_size = (1, ) + input_shape[3:]
        return F.avg_pool3d(input, kernel_size, self.stride,
                            self.padding, self.ceil_mode, self.count_include_pad)