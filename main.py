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

from lib.dataset import VideoDataSet
from lib.models import VideoModule
from lib.transforms import *
from lib.utils.tools import *
from lib.opts import args

from train_val import train, validate

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
                             "data/{}".format(args.dataset))

    # create model
    org_model = VideoModule(num_class=num_class, 
        base_model_name=args.arch,
        t_length=args.t_length,
        t_stride=args.t_stride,
        dropout=args.dropout,
        pretrained=args.pretrained)

    model = torch.nn.DataParallel(org_model).cuda()

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss().cuda()

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
    train_transform = torchvision.transforms.Compose([
        org_model.get_augmentation(),
        Stack(mode=args.mode),
        ToTorchFormatTensor(),
        GroupNormalize(),
        ])
    train_dataset = VideoDataSet(root_path=data_root, 
        list_file=args.train_list,
        t_length=args.t_length, 
        t_stride=args.t_stride, 
        image_tmpl="image_{:06d}.jpg", 
        transform=train_transform,
        phase="Train")
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.workers, pin_memory=True)

    ## val data
    val_transform = torchvision.transforms.Compose([
        GroupScale(256),
        GroupCenterCrop(224),
        Stack(mode=args.mode),
        ToTorchFormatTensor(),
        GroupNormalize(),
        ])
    val_dataset = VideoDataSet(root_path=data_root, 
        list_file=args.val_list,
        t_length=args.t_length,
        t_stride=args.t_stride,
        image_tmpl="image_{:06d}.jpg",
        transform=val_transform,
        phase="Val")
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False, 
        num_workers=args.workers, pin_memory=True)

    if args.mode != "3D":
        cudnn.benchmark = True

    validate(val_loader, model, criterion, args.print_freq, 0)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr_steps)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args.print_freq)

        # evaluate on validation set
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            metric = validate(val_loader, model, criterion, args.print_freq, epoch + 1)

            # remember best prec@1 and save checkpoint
            is_best = metric > best_metric
            best_metric = max(metric, best_metric)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_metric': best_metric,
                'optimizer': optimizer.state_dict(),
            }, is_best, epoch + 1)

if __name__ == '__main__':
    main()
