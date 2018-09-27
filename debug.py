import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

cudnn.benchmark = True

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

input = input.cuda()
layer.cuda()

output = layer(input)
