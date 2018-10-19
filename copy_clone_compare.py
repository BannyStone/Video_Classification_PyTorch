import torch
import ipdb

'''
# First, let's try torch.Tensor.copy_()
va = torch.ones(2, 3).requires_grad_()
vb = 2 * va
torch.autograd.backward(vb, torch.ones(2, 3)) # va.grad= [2...]
vc = torch.Tensor(2, 3)
vc.copy_(va)
vd = 2 * vc
torch.autograd.backward(vd, torch.ones(2, 3)) # va.grad = [4...]
'''

'''
# Second, let's try torch.Tensor.cuda()
va = torch.ones(2, 3).requires_grad_()
vb = 2 * va
torch.autograd.backward(vb, torch.ones(2, 3))
vc = va.cuda(0)
ipdb.set_trace()
vd = 2 * vc
torch.autograd.backward(vd, torch.ones(2, 3, device=0))
ipdb.set_trace()
'''

# Third, let's try torch.Tensor.clone()
va = torch.ones(2, 3).requires_grad_()
vb = 2 * va
torch.autograd.backward(vb, torch.ones(2, 3)) # va.grad= [2...]
vc = va.clone()
ipdb.set_trace()
vd = 2 * vc
torch.autograd.backward(vd, torch.ones(2, 3)) # va.grad = [4...]
ipdb.set_trace()