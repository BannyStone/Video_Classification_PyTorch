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

from .module import FlexModule
import torch
from torch.nn import functional as F

class _ConvNd(FlexModule):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias, parameters):
        assert('weight' in parameters), "conv parameters must contain weight"
        if self.bias:
            assert('bias' in parameters), "when bias is true, conv parameters must contain bias"
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

        # load weight
        weight_tensor = parameters['weight']
        assert(weight_tensor.is_leaf == False and weight_tensor.require_grad == True), "Weight tensor cannot be leaf."
        if transposed:
            assert(weight_tensor.shape == (self.in_channels, self.out_channels // self.groups, *self.kernel_size)), 
                    "weight tensor is not compatible with configuration"
        else:
            assert(weight_tensor.shape == (self.out_channels, self.in_channels // self.groups, *self.kernel_size)), 
                    "weight tensor is not compatible with configuration"
        self.weight = weight_tensor

        # load bias conditionally
        if bias:
            assert(bias_tensor.is_leaf == False and bias_tensor.require_grad == True), "Bias tensor cannot be leaf."
            assert(bias_tensor.shape == (out_channels,)), "bias tensor is not compatible with configuration"
            self.bias = bias_tensor
        else:
            self.bias = None

        # register params
        self.register_parameter("weight", self.weight)
        self.register_parameter("bias", self.bias)

class Conv2d(_ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, parameters=None):
        assert(parameters is not None), 
               'In Flexible Modules, attr \"parameters\" cannot be none.'
        assert('weight' in parameters), 
               'In Flexible Modules, attr \"parameters\" should contain key \"weight\"'
        if bias:
            assert('bias' in parameters), 
                   'In Flexible Modules, attr \"parameters\" should contain key \"bias\"'
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, parameters)

    def forward(self, input):
        return F.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
