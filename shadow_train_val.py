import os
import time
import logging

import torch

from lib.utils.tools import *
import ipdb

def train_bk(train_loader, model, criterion, optimizer, epoch, print_freq):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
      # measure data loading time
      data_time.update(time.time() - end)
      ipdb.set_trace()

      # input = input.cuda(non_blocking=True) # comment when using dataparallel
      target = target.cuda(non_blocking=True)

      # compute output
      # ipdb.set_trace()
      output = model(input)
      # ipdb.set_trace()
      loss = criterion(output, target)

      # measure accuracy and record loss
      prec1, prec5 = accuracy(output, target, topk=(1, 5))
      losses.update(loss.item(), input.size(0))
      top1.update(prec1.item(), input.size(0))
      top5.update(prec5.item(), input.size(0))

      # compute gradient and do SGD step
      optimizer.zero_grad()
      loss.backward()
      from IPython import embed
      embed()
      # ipdb.set_trace()
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

def train(train_loader, model, criterion, optimizer, epoch, print_freq):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    top1_2 = AverageMeter()
    top5_2 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
      # measure data loading time
      data_time.update(time.time() - end)

      # input = input.cuda(non_blocking=True) # comment when using dataparallel
      target = target.cuda(non_blocking=True)

      # compute output
      output1, output2 = model(input)
      loss1 = criterion(output1, target)
      loss2 = criterion(output2, target)
      loss = loss1 + loss2

      # measure accuracy and record loss
      prec1, prec5 = accuracy(output1, target, topk=(1, 5))
      prec1_2, prec5_2 = accuracy(output2, target, topk=(1, 5))
      losses.update(loss.item(), input.size(0))
      top1.update(prec1.item(), input.size(0))
      top5.update(prec5.item(), input.size(0))
      top1_2.update(prec1_2.item(), input.size(0))
      top5_2.update(prec5_2.item(), input.size(0))

      # compute gradient and do SGD step
      optimizer.zero_grad()
      loss1.backward()
      # from IPython import embed
      # embed()
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
                'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                'Prec_2@1 {top1_2.val:.3f} ({top1_2.avg:.3f})\t'
                'Prec_2@5 {top5_2.val:.3f} ({top5_2.avg:.3f})\t'.format(
                 epoch, i, len(train_loader), batch_time=batch_time,
                 data_time=data_time, loss=losses, top1=top1, 
                 top5=top5, top1_2=top1_2, top5_2=top5_2, 
                 lr=optimizer.param_groups[-1]['lr'])))


def validate(val_loader, model, criterion, print_freq, epoch, logger=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
      # print("pass")
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
          # input = input.cuda(non_blocking=True) # comment when using dataparallel
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

    return (top1.avg + top5.avg) / 2