import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import ipdb
# cudnn.benchmark = True

# Bad
# input = torch.ones((4, 512, 4, 7, 7))
# layer = nn.Conv3d(512, 2047, kernel_size=1, bias=False)

# Good
# input = torch.ones((4, 512, 4, 7, 7))
# layer = nn.Conv3d(512, 2048, kernel_size=1, bias=False)

# 1800 good

# Bad
# input = torch.ones((4, 1024, 4, 7, 7))
# layer = nn.Conv3d(1024, 1024, kernel_size=1, bias=False)

# Good
# input = torch.ones((4, 1024, 4, 7, 7))
# layer = nn.Conv3d(1024, 512, kernel_size=1, bias=False)

# Bad
# input = torch.ones((4, 2048, 4, 7, 7))
# layer = nn.Conv3d(2048, 512, kernel_size=1, bias=False)

# input = torch.ones((4, 8192, 56, 56))
# layer = nn.Conv2d(8192, 8192, kernel_size=1, bias=False)

# input = input.cuda()
# layer.cuda()

# output = layer(input)

# va = torch.ones(2,3)
# va.requires_grad_()
# vb = torch.ones(2,3).mul(2)
# vb.requires_grad_()
# ipdb.set_trace()
# va.copy_(vb.mul(4))
# # vc = vb.mul(4)
# ipdb.set_trace()

# First, let's try torch.Tensor.copy_()
# va = torch.ones(2, 3)# .requires_grad_()
# va100 = va.mul(100)
# # torch.autograd.backward(vb, torch.ones(2, 3))
# # ipdb.set_trace()
# vc = torch.ones(2, 3).requires_grad_()
# vc10 = vc.mul(10)
# ipdb.set_trace()
# va.copy_(vc10)
# # vd = 2 * vc
# # torch.autograd.backward(vd, torch.ones(2, 3))
# ipdb.set_trace()
# va.copy_(vc)
# ipdb.set_trace()

# data parallel debugging
import os
os.environ['CUDA_VISIBLE_DEVICES']='3,5'
class TestModel(nn.Module):
	def __init__(self):
		super(TestModel, self).__init__()
		self.layer1 = nn.Conv2d(3, 64, 3, bias=False)
		self.layer2 = nn.Conv2d(64, 128, 3, bias=False)
		self.layer3 = nn.Sequential(nn.Conv2d(128,128,3,bias=False), nn.Conv2d(128,2,3, bias=False))
	def forward(self, x):
		print("layer1", self.layer1.weight.is_leaf, self.layer1.weight.requires_grad)
		print("layer2", self.layer2.weight.is_leaf, self.layer2.weight.requires_grad)
		out = self.layer1(x)
		out = self.layer2(out)
		out = self.layer3(out)
		return out

org_model = TestModel()
model = torch.nn.DataParallel(org_model).cuda()
ipdb.set_trace()

x = torch.ones(8, 3, 56, 56)
out = model(x)
