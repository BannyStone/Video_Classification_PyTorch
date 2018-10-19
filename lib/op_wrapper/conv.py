import collections
from itertools import repeat

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)

import math
from .module import FlexModule
import torch
from torch.nn import functional as F
from torch.nn import init

class _ConvNd(FlexModule):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias):
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        if transposed:
            weight_tensor = torch.Tensor(
                in_channels, out_channels // groups, *kernel_size)
        else:
            weight_tensor = torch.Tensor(
                out_channels, in_channels // groups, *kernel_size)
        self.register_parameter('weight', weight_tensor)
        if bias:
            bias_tensor = torch.Tensor(out_channels)
        else:
            bias_tensor = None
        self.register_parameter('bias', bias_tensor)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)

class Conv2d(_ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)

    def forward(self, input):
        return F.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class Conv3d(_ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)
        dilation = _triple(dilation)
        super(Conv3d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _triple(0), groups, bias)

    def forward(self, input):
        return F.conv3d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
# class _ConvNd(FlexModule):

#     def __init__(self, in_channels, out_channels, kernel_size, stride,
#                  padding, dilation, transposed, output_padding, groups, bias, parameters):
#         assert('weight' in parameters), "conv parameters must contain weight"
#         if self.bias:
#             assert('bias' in parameters), "when bias is true, conv parameters must contain bias"
#         super(_ConvNd, self).__init__()
#         if in_channels % groups != 0:
#             raise ValueError('in_channels must be divisible by groups')
#         if out_channels % groups != 0:
#             raise ValueError('out_channels must be divisible by groups')
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding
#         self.dilation = dilation
#         self.transposed = transposed
#         self.output_padding = output_padding
#         self.groups = groups

#         # load weight
#         weight_tensor = parameters['weight']
#         assert(weight_tensor.is_leaf == False and weight_tensor.require_grad == True), "Weight tensor cannot be leaf."
#         if transposed:
#             assert(weight_tensor.shape == (self.in_channels, self.out_channels // self.groups, *self.kernel_size)), 
#                     "weight tensor is not compatible with configuration"
#         else:
#             assert(weight_tensor.shape == (self.out_channels, self.in_channels // self.groups, *self.kernel_size)), 
#                     "weight tensor is not compatible with configuration"
#         self.weight = weight_tensor

#         # load bias conditionally
#         if bias:
#             assert(bias_tensor.is_leaf == False and bias_tensor.require_grad == True), "Bias tensor cannot be leaf."
#             assert(bias_tensor.shape == (out_channels,)), "bias tensor is not compatible with configuration"
#             self.bias = bias_tensor
#         else:
#             self.bias = None

#         # register params
#         self.register_parameter("weight", self.weight)
#         self.register_parameter("bias", self.bias)

# class Conv2d(_ConvNd):

#     def __init__(self, in_channels, out_channels, kernel_size, stride=1,
#                  padding=0, dilation=1, groups=1, bias=True, parameters=None):
#         assert(parameters is not None), 
#                'In Flexible Modules, attr \"parameters\" cannot be none.'
#         assert('weight' in parameters), 
#                'In Flexible Modules, attr \"parameters\" should contain key \"weight\"'
#         if bias:
#             assert('bias' in parameters), 
#                    'In Flexible Modules, attr \"parameters\" should contain key \"bias\"'
#         kernel_size = _pair(kernel_size)
#         stride = _pair(stride)
#         padding = _pair(padding)
#         dilation = _pair(dilation)
#         super(Conv2d, self).__init__(
#             in_channels, out_channels, kernel_size, stride, padding, dilation,
#             False, _pair(0), groups, bias, parameters)

#     def forward(self, input):
#         return F.conv2d(input, self.weight, self.bias, self.stride,
#                         self.padding, self.dilation, self.groups)
