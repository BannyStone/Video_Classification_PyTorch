import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.model_zoo as model_zoo
import math

class adaTD(nn.Module):
	def __init__(self, target_T):
		super(adaTD, self).__init__()
		self.target_T = target_T
		self.ceil_mode = False
        self.count_include_pad = True

	def forward(self, x):
		stride = x.shape[2] // self.target_T
		kernel_size = stride
		return F.max_pool3d(x, kernel_size=(kernel_size, 1, 1),
							stride=(stride, 1, 1),
                            padding=self.padding,
                            ceil_mode=self.ceil_mode)

class sConv(nn.Module):
	def __init__(self, inplanes, planes, s_kernel_size=3, s_stride=1):
		