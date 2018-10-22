from .module import FlexModule
import torch
from torch.nn import functional as F
from torch.nn import init

class _BatchNorm(FlexModule):
    _version = 2

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(_BatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        # register weight and bias
        self.shapes['weight'] = (num_features,) if affine else None
        self.shapes['bias'] = (num_features,) if affine else None
        self.shapes['running_mean'] = (num_features,) if track_running_stats else None
        self.shapes['running_var'] = (num_features,) if track_running_stats else None
        self.shapes['num_batches_tracked'] = (1,) if track_running_stats else None
        self.register_nonleaf_parameter('weight', None)
        self.register_nonleaf_parameter('bias', None)
        self.register_buffer('running_mean', None)
        self.register_buffer('running_var', None)
        self.register_buffer('num_batches_tracked', None)

    def _check_input_dim(self, input):
        raise NotImplementedError

    def forward(self, input):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats,
            exponential_average_factor, self.eps)

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

class BatchNorm2d(_BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

class BatchNorm3d(_BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))
