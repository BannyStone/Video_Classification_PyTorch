import math
from .module import FlexModule
import torch
from torch.nn import functional as F
from torch.nn import init

class Linear(FlexModule):

    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_parameter('weight', torch.Tensor(out_features, in_features))
        if bias:
            self.register_parameter('bias', torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

# class Linear(FlexModule):

#     def __init__(self, in_features, out_features, bias=True, parameters=None):
#         weight_tensor = parameters['weight']
#         if bias:
#             bias_tensor = parameters['bias']
#         assert(weight_tensor is not None), "In Flexible Modules, wieght tensor cannot be none."
#         super(Linear, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         assert(weight_tensor.shape == (out_features, in_features)), 
#                "Weight shape is not compatible with configuration"
#         self.weight = weight_tensor
#         self.register_parameter('weight', self.weight)
#         if bias:
#             assert(bias_tensor.shape == (out_features)),
#                    "Bias shape is not compatible with configuration"
#             self.bias = bias_tensor
#         else:
#             self.bias = None
#         self.register_parameter('bias', self.bias)

#     def forward(self, input):
#         return F.linear(input, self.weight, self.bias)
