from lib.networks.part_inflate_resnet_3d import *
from lib.modules import *
import torch
from lib.networks.km_resnet_3d_beta import TKMConv, compute_tkmconv, km_resnet26_3d_v2_sample, km_resnet50_3d_v2_sample
def count_GloAvgPool3d(m, x, y):
	m.total_ops = torch.Tensor([int(0)])

from thop import profile
model = km_resnet50_3d_v2_sample()
model.fc = torch.nn.Linear(2048, 400)
flops, params = profile(model, input_size=(1, 3, 8, 224,224), custom_ops={GloAvgPool3d: count_GloAvgPool3d, TKMConv: compute_tkmconv})
print("params: {}".format(params/1000000))
print("flops: {}".format(flops/1000000000))
