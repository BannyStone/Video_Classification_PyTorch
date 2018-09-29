import argparse
import time

import numpy as np
import torch.nn.parallel
import torch.optim
from sklearn.metrics import confusion_matrix

from lib.dataset import VideoDataSet
from lib.models import VideoModule
from lib.transforms import *
from lib.utils.tools import *

# options
parser = argparse.ArgumentParser(
    description="Standard video-level testing")
parser.add_argument('dataset', type=str, choices=['ucf101', 'hmdb51', 'kinetics400', 'kinetics200'])
parser.add_argument('test_list', type=str)
parser.add_argument('weights', type=str)
parser.add_argument('--arch', type=str, default="resnet101")
parser.add_argument('--save_scores', type=str, default=None)
parser.add_argument('--num_clips', type=int, default=20)
parser.add_argument('--max_num', type=int, default=-1)
parser.add_argument('--test_crops', type=int, default=10)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--crop_fusion_type', type=str, default='avg',
                    choices=['avg', 'max', 'topk'])
parser.add_argument('--k', type=int, default=3)
parser.add_argument('--dropout', type=float, default=0.7)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--gpus', nargs='+', type=int, default=None)

args = parser.parse_args()


if args.dataset == 'ucf101':
    num_class = 101
elif args.dataset == 'hmdb51':
    num_class = 51
elif args.dataset == 'kinetics400':
    num_class = 400
elif args.dataset == 'kinetics200':
    num_class = 200
else:
    raise ValueError('Unknown dataset '+args.dataset)

net = VideoModule(num_class=num_class, 
                  base_model_name=args.arch,
                  t_length=args.t_length,
                  t_stride=args.t_stride,
                  dropout=args.dropout,)
num_params = 0
for param in org_model.parameters():
    num_params += param.reshape((-1, 1)).shape[0]
print("Model Size is {:.3f}M".format(num_params/1000000))
# model = torch.nn.DataParallel(org_model).cuda()

## test data
test_transform = torchvision.transforms.Compose([
    GroupScale(256),
    GroupCenterCrop(224),
    Stack(mode=args.mode),
    ToTorchFormatTensor(),
    GroupNormalize(),
    ])
test_dataset = VideoDataSet(root_path=data_root, 
    list_file=args.test_list,
    t_length=args.t_length,
    t_stride=args.t_stride,
    image_tmpl="image_{:06d}.jpg",
    transform=val_transform,
    phase="Val")
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=args.batch_size, shuffle=False, 
    num_workers=args.workers, pin_memory=True)