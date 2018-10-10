import math
from .module import FlexModule
import torch
from torch.nn import functional as F

class Linear(FlexModule):

    def __init__(self, in_features, out_features, bias=True, parameters=None):
        weight_tensor = parameters['weight']
        if bias:
            bias_tensor = parameters['bias']
        assert(weight_tensor is not None), "In Flexible Modules, wieght tensor cannot be none."
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        assert(weight_tensor.shape == (out_features, in_features)), 
               "Weight shape is not compatible with configuration"
        self.weight = weight_tensor
        self.register_parameter('weight', self.weight)
        if bias:
            assert(bias_tensor.shape == (out_features)),
                   "Bias shape is not compatible with configuration"
            self.bias = bias_tensor
        else:
            self.bias = None
        self.register_parameter('bias', self.bias)

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)
