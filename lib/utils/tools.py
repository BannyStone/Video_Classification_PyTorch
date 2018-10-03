import os
import numpy as np
import logging
import torch
import shutil

__all__ = ['AverageMeter', 'save_checkpoint', 'adjust_learning_rate', 'accuracy']

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(state, is_best, epoch, experiment_root, filename='checkpoint_{}epoch.pth'):
    filename = os.path.join(experiment_root, filename.format(epoch))
    logging.info("saving model to {}...".format(filename))
    torch.save(state, filename)
    if is_best:
        best_name = os.path.join(experiment_root, 'model_best.pth')
        shutil.copyfile(filename, best_name)
    logging.info("saving done.")

def adjust_learning_rate(optimizer, base_lr, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    lr = base_lr * decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res