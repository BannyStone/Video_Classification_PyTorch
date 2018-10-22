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
        # self.shapes = {}
        self.shapes['weight'] = (out_features, in_features)
        self.shapes['bias'] = (out_features,) if bias else None
        self.register_nonleaf_parameter('weight', None)
        self.register_nonleaf_parameter('bias', None)

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
