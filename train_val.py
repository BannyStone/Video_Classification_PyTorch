import os
import time
import logging

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

from lib.utils.tools import *

def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
      m.eval()

def train(train_loader, model, criterion, optimizer, epoch, print_freq, scheduler):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
      scheduler.step()
      # measure data loading time
      data_time.update(time.time() - end)

      # input = input.cuda()
      target = target.cuda(non_blocking=True)

      # compute output
      output = model(input)
      loss = criterion(output, target)

      # measure accuracy and record loss
      prec1, prec5 = accuracy(output, target, topk=(1, 5))
      losses.update(loss.item(), input.size(0))
      top1.update(prec1.item(), input.size(0))
      top5.update(prec5.item(), input.size(0))

      # compute gradient and do SGD step
      optimizer.zero_grad()
      loss.backward()
      # clip gradients
      # total_norm = clip_grad_norm_(model.parameters(), 20)
      # # print(total_norm)
      # if total_norm > 20:
      #   print("clipping gradient: {} with coef {}".format(total_norm, 20 / total_norm))

      optimizer.step()

      # measure elapsed time
      batch_time.update(time.time() - end)
      end = time.time()

      if i % print_freq == 0:
          logging.info(('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                'Batch {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                 epoch, i, len(train_loader), batch_time=batch_time,
                 data_time=data_time, loss=losses, top1=top1, 
                 top5=top5, lr=optimizer.param_groups[-1]['lr'])))

def finetune(train_loader, model, criterion, optimizer, epoch, print_freq):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()

    # switch mode
    for m in model.modules():
      if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        m.eval()
      if isinstance(m, nn.Dropout):
        m.eval()

    # block gradients to base model
    for param in model.named_parameters():
      if "base_model" in param[0]:
        param[1].requires_grad = False

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
      # import pdb
      # pdb.set_trace()
      # print("conv1", model.state_dict()['module.base_model.conv1.weight'].view(-1)[0:3])
      # print("fc", model.state_dict()['module.classifier.1.weight'].view(-1)[0:3])
      # print(model.state_dict().view(-1)[0:3])
      # measure data loading time
      data_time.update(time.time() - end)

      # input = input.cuda()
      target = target.cuda(non_blocking=True)

      # compute output
      output = model(input)
      loss = criterion(output, target)

      # measure accuracy and record loss
      prec1, prec5 = accuracy(output, target, topk=(1, 5))
      losses.update(loss.item(), input.size(0))
      top1.update(prec1.item(), input.size(0))
      top5.update(prec5.item(), input.size(0))

      # compute gradient and do SGD step
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      # measure elapsed time
      batch_time.update(time.time() - end)
      end = time.time()

      if i % print_freq == 0:
          logging.info(('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                'Batch {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                 epoch, i, len(train_loader), batch_time=batch_time,
                 data_time=data_time, loss=losses, top1=top1, 
                 top5=top5, lr=optimizer.param_groups[-1]['lr'])))

def finetune_new(train_loader, model, criterion, optimizer, epoch, print_freq):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()

    # model.apply(set_bn_eval)

    # switch mode
    for n, m in model.named_modules():
      if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        # m.eval()
        if "base_model.bn1" in n:
          print(n)
          pass
        else:
          for p in m.parameters():
            p.requires_grad = False
          m.eval()

    # for n, m in model.named_modules():
    #   if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
    #       m.eval()
      # if isinstance(m, nn.Dropout):
      #   m.eval()

    # block gradients to base model
    # for param in model.named_parameters():
    #   if "bn" in param[0]:
    #     print(param[1].requires_grad)
    #   if "base_model" in param[0]:
    #     param[1].requires_grad = False

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
      # print(model.module.base_model.bn1.weight.view(-1)[:3])
      # print(model.module.base_model.bn1.running_mean.view(-1)[:3])
      # import pdb
      # pdb.set_trace()
      # print("conv1", model.state_dict()['module.base_model.conv1.weight'].view(-1)[0:3])
      # print("fc", model.state_dict()['module.classifier.1.weight'].view(-1)[0:3])
      # print(model.state_dict().view(-1)[0:3])
      # measure data loading time
      data_time.update(time.time() - end)

      # input = input.cuda()
      target = target.cuda(non_blocking=True)

      # compute output
      output = model(input)
      loss = criterion(output, target)

      # measure accuracy and record loss
      prec1, prec5 = accuracy(output, target, topk=(1, 5))
      losses.update(loss.item(), input.size(0))
      top1.update(prec1.item(), input.size(0))
      top5.update(prec5.item(), input.size(0))

      # compute gradient and do SGD step
      optimizer.zero_grad()
      loss.backward()
      # for param in model.parameters():
      #   param.grad.data.clamp_(-1, 1)
      total_norm = clip_grad_norm(model.parameters(), 20)
      if total_norm > 20:
        print("clipping gradient: {} with coef {}".format(total_norm, 20 / total_norm))

      optimizer.step()

      # measure elapsed time
      batch_time.update(time.time() - end)
      end = time.time()

      if i % print_freq == 0:
          logging.info(('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                'Batch {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                 epoch, i, len(train_loader), batch_time=batch_time,
                 data_time=data_time, loss=losses, top1=top1, 
                 top5=top5, lr=optimizer.param_groups[-1]['lr'])))

def validate(val_loader, model, criterion, print_freq, epoch, logger=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                logging.info(('Test: [{0}/{1}]\t'
                      'Batch {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5)))

    logging.info(('Epoch {epoch} Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
          .format(epoch=epoch, top1=top1, top5=top5, loss=losses)))

    # return (top1.avg + top5.avg) / 2
    return top1.avg