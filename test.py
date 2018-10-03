import argparse
import time
import os
import numpy as np
import torch.nn.parallel
import torch.optim
# from sklearn.metrics import confusion_matrix

from lib.dataset import VideoDataSet
from lib.models import VideoModule, TSN
from lib.transforms import *
from lib.utils.tools import AverageMeter, accuracy

import pdb

# options
parser = argparse.ArgumentParser(
    description="Standard video-level testing")
parser.add_argument('dataset', type=str, choices=['ucf101', 'hmdb51', 'kinetics400', 'kinetics200'])
parser.add_argument('test_list', type=str)
parser.add_argument('weights', type=str)
parser.add_argument('--arch', type=str, default="resnet50_3d_lite")
parser.add_argument('--mode', type=str, default="TSN+3D")
parser.add_argument('--save_scores', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--num_segments', type=int, default=20)
parser.add_argument('--test_crops', type=int, default=10)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--t_length', type=int, default=8)
parser.add_argument('--t_stride', type=int, default=8)
parser.add_argument('--crop_fusion_type', type=str, default='avg',
                    choices=['avg', 'max', 'topk'])
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

args = parser.parse_args()

def main():
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

    net = VideoModule(num_class=num_class, 
                      base_model_name=args.arch,
                      dropout=args.dropout, 
                      pretrained=False)
    
    # compute params number of a model
    num_params = 0
    for param in net.parameters():
        num_params += param.reshape((-1, 1)).shape[0]
    print("Model Size is {:.3f}M".format(num_params / 1000000))

    net = torch.nn.DataParallel(net).cuda()

    # load weights
    model_state = torch.load(args.weights)
    state_dict = model_state['state_dict']
    test_epoch = model_state['epoch']
    arch = model_state['arch']
    assert arch == args.arch
    net.load_state_dict(state_dict)
    tsn = TSN(args.batch_size, net, 
              args.num_segments, args.t_length, 
              crop_fusion_type=args.crop_fusion_type, 
              mode=args.mode).cuda()

    ## test data
    test_transform = torchvision.transforms.Compose([
        GroupOverSample(args.input_size, 256),
        Stack(mode=args.mode),
        ToTorchFormatTensor(),
        GroupNormalize(),
        ])
    test_dataset = VideoDataSet(
        root_path=data_root, 
        list_file=args.test_list,
        t_length=args.t_length,
        t_stride=args.t_stride,
        num_segments=args.num_segments,
        image_tmpl="image_{:06d}.jpg",
        transform=test_transform,
        phase="Test")
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    
    # Test
    batch_timer = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    results = None

    # set eval mode
    tsn.eval()

    end = time.time()
    for ind, (data, label) in enumerate(test_loader):
        label = label.cuda(non_blocking=True)

        with torch.no_grad():
            output, pred = tsn(data)
            prec1, prec5 = accuracy(pred, label, topk=(1, 5))
            top1.update(prec1.item(), data.shape[0])
            top5.update(prec5.item(), data.shape[0])

        # pdb.set_trace()
        batch_timer.update(time.time() - end)
        end = time.time()
        if results is not None:
            np.concatenate((results, output.cpu().numpy()), axis=0)
        else:
            results = output.cpu().numpy()
        print("{0}/{1} done, Batch: {batch_timer.val:.3f}({batch_timer.avg:.3f}), \
              Top1: {top1.val:>6.3f}({top1.avg:>6.3f}), \
              Top5: {top5.val:>6.3f}({top5.avg:>6.3f})".
              format(ind + 1, len(test_loader), 
                batch_timer=batch_timer, 
                top1=top1, top5=top5))
    target_file = os.path.join(args.save_scores, "arch_{0}-epoch_{1}-top1_{2}-top5_{3}.npz".format(arch, test_epoch, top1.avg, top5.avg))
    print("saving {}".format(target_file))
    np.savez(target_file, results)
if __name__ == "__main__":
    main()