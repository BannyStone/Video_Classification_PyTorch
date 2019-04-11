import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

class Scale2d(nn.Module):
    def __init__(self, out_channels):
        super(Scale2d, self).__init__()
        self.scale = Parameter(torch.Tensor(1, out_channels, 1, 1))

    def forward(self, input):
        return input * self.scale

class Scale3d(nn.Module):
    def __init__(self, out_channels):
        super(Scale3d, self).__init__()
        self.scale = Parameter(torch.Tensor(1, out_channels, 1, 1, 1))

    def forward(self, input):
        return input * self.scale