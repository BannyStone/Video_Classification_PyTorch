from .module import FlexModule
import torch
from torch.nn import functional as F

class _BatchNorm(FlexModule):

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, parameters=None):
        assert(parameters is not None), "In Flexible Modules, parameters cannot be none."
        assert('weight' in parameters and 'bias' in parameters 
               and 'running_mean' in parameters and 'running_var' in parameters
               and 'num_batches_tracked' in parameters), "BatchNorm should have 5 attrs."
        weight_tensor = parameters['weight']
        bias_tensor = parameters['bias']
        mean_tensor = parameters['running_mean']
        var_tensor = parameters['running_var']
        num_batches_tracked = parameters['num_batches_tracked']
        assert(weight_tensor and bias_tensor and mean_tensor and var_tensor and num_batches_tracked is not None),
                "In Flexible Modules, parameter and buffer tensors must be specified."
        super(_BatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        assert(weight_tensor.is_leaf == False and weight_tensor.require_grad == True), "Weight tensor cannot be leaf."
        assert(bias_tensor.is_leaf == False and bias_tensor.require_grad == True), "Bias tensor cannot be leaf."
        if self.affine:
            assert(weight_tensor.shape == (num_features,)), "Weight tensor is not compatible with configuration"
            assert(bias_tensor.shape == (num_features,)), "Bias tensor is not compatible with configuration"
            self.weight = weight_tensor
            self.bias = bias_tensor
        else:
            self.weight = None
            self.bias = None
        self.register_parameter('weight', self.weight)
        self.register_parameter('bias',self.bias)
        if self.track_running_stats:
            assert(mean_tensor.shape == (num_features,)), "Mean tensor is not compatible with configuration"
            assert(var_tensor.shape == (num_features,)), "Var tensor is not compatible with configuration"
            assert(num_batches_tracked.numel() == 1), "Num batches tracked must be a tensor with one element"
            self.running_mean = mean_tensor
            self.running_var = var_tensor
            self.num_batches_tracked = num_batches_tracked
        else:
            self.running_mean = None
            self.running_var = None
            self.num_batches_tracked = None
        self.register_buffer('running_mean', self.running_mean)
        self.register_buffer('running_var', self.running_var)
        self.register_buffer('num_batches_tracked', self.num_batches_tracked)

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

    def _load_from_state_dict(self, state_dict, prefix, metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = metadata.get('version', None)

        if (version is None or version < 2) and self.track_running_stats:
            # at version 2: added num_batches_tracked buffer
            #               this should have a default value of 0
            num_batches_tracked_key = prefix + 'num_batches_tracked'
            if num_batches_tracked_key not in state_dict:
                state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)

        super(_BatchNorm, self)._load_from_state_dict(
            state_dict, prefix, metadata, strict,
            missing_keys, unexpected_keys, error_msgs)


class BatchNorm2d(_BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))
