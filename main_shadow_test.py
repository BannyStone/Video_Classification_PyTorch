import argparse
import os
import time
import shutil
import logging

import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.parameter import Parameter

from lib.dataset import VideoDataSet, VideoDebugDataSet
from lib.models import VideoModule, VideoShadowModule
from lib.transforms import *
from lib.utils.tools import *
from lib.opts import args

from shadow_train_val import train, validate

import ipdb

best_metric = 0

def main():
    global args, best_metric

    # specify dataset
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

    data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                             "data/{}/access".format(args.dataset))

    # create model
    org_model = VideoShadowModule(num_class=num_class,
        base_model_name=args.arch,
        dropout=args.dropout,
        pretrained=args.pretrained,
        pretrained_model=args.pretrained_model)
    num_params = 0
    for param in org_model.parameters():
        if isinstance(param, Parameter):
            num_params += param.reshape((-1, 1)).shape[0]
    print("Model Size is {:.3f}M".format(num_params/1000000))

    model = torch.nn.DataParallel(org_model).cuda()
    # model = org_model.cuda()

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss().cuda()

    # ipdb.set_trace()

    optimizer = torch.optim.SGD(model.parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_metric = checkpoint['best_metric']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))

    # Data loading code
    ## train data
    # train_transform = torchvision.transforms.Compose([
    #     org_model.get_augmentation(),
    #     Stack(mode=args.mode),
    #     ToTorchFormatTensor(),
    #     GroupNormalize(),
    #     ])
    train_dataset = VideoDebugDataSet()
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.workers, pin_memory=True)

    if args.mode != "3D":
        cudnn.benchmark = True

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, args.lr, epoch, args.lr_steps)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args.print_freq)

if __name__ == '__main__':
    main()
