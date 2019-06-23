import argparse
import time
import os
import numpy as np
import torch.nn.parallel
import torch.optim
# from sklearn.metrics import confusion_matrix

from lib.dataset import VideoDataSet, ShortVideoDataSet
from lib.models import VideoModule, TSN
from lib.transforms import *
from lib.utils.tools import AverageMeter, accuracy

import pdb
import logging

def set_logger(debug_mode=False):
    import time
    from time import gmtime, strftime
    logdir = os.path.join(args.experiment_root, 'log')
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    log_file = "logfile_" + time.strftime("%d_%b_%Y_%H:%M:%S", time.localtime())
    log_file = os.path.join(logdir, log_file)
    handlers = [logging.FileHandler(log_file), logging.StreamHandler()]

    """ add '%(filename)s:%(lineno)d %(levelname)s:' to format show source file """
    logging.basicConfig(level=logging.DEBUG if debug_mode else logging.INFO,
                format='%(asctime)s: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
                handlers = handlers)

# options
parser = argparse.ArgumentParser(
    description="Standard video-level testing")
parser.add_argument('dataset', type=str)
parser.add_argument('test_list', type=str)
parser.add_argument('weights', type=str)
parser.add_argument('--arch', type=str, default="resnet50_3d_v1")
parser.add_argument('--mode', type=str, default="TSN+3D")
# parser.add_argument('--save_scores', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--num_segments', type=int, default=10)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--resize', type=int, default=256)
parser.add_argument('--t_length', type=int, default=16)
parser.add_argument('--t_stride', type=int, default=4)
# parser.add_argument('--crop_fusion_type', type=str, default='avg',
#                     choices=['avg', 'max', 'topk'])
parser.add_argument('--image_tmpl', type=str)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

args = parser.parse_args()

experiment_id = '_'.join(map(str, ['test', args.dataset, args.arch, args.mode, 
           'length'+str(args.t_length), 'stride'+str(args.t_stride), 
           'seg'+str(args.num_segments)]))

args.experiment_root = os.path.join('./output', experiment_id)

set_logger()
logging.info(args)
if not os.path.exists(args.experiment_root):
     os.makedirs(args.experiment_root)

def main():
    if args.dataset == 'ucf101':
        num_class = 101
    elif args.dataset == 'hmdb51':
        num_class = 51
    elif args.dataset == 'kinetics400':
        num_class = 400
    elif args.dataset == 'kinetics200':
        num_class = 200
    elif args.dataset == 'sthsth_v1':
        num_class = 174
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
    logging.info("Model Size is {:.3f}M".format(num_params / 1000000))

    net = torch.nn.DataParallel(net).cuda()
    net.eval()

    # load weights
    model_state = torch.load(args.weights)
    state_dict = model_state['state_dict']
    test_epoch = model_state['epoch']
    best_metric = model_state['best_metric']
    arch = model_state['arch']
    logging.info("Model Epoch: {}; Best_Top1: {}".format(test_epoch, best_metric))
    assert arch == args.arch
    net.load_state_dict(state_dict)
    tsn = TSN(args.batch_size, net, 
              args.num_segments, args.t_length, 
              mode=args.mode).cuda()

    ## test data
    test_transform = torchvision.transforms.Compose([
        GroupScale(256),
        GroupOverSampleKaiming(args.input_size),
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
        image_tmpl=args.image_tmpl,
        transform=test_transform,
        phase="Test")
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    
    # Test
    batch_timer = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()
    top1_a = AverageMeter()
    top5_a = AverageMeter()
    results_m = None
    results_a = None

    # set eval mode
    tsn.eval()

    end = time.time()
    for ind, (data, label) in enumerate(test_loader):
        label = label.cuda(non_blocking=True)

        with torch.no_grad():
            output_m, pred_m, output_a, pred_a = tsn(data)
            prec1_m, prec5_m = accuracy(pred_m, label, topk=(1, 5))
            prec1_a, prec5_a = accuracy(pred_a, label, topk=(1, 5))
            top1_m.update(prec1_m.item(), data.shape[0])
            top5_m.update(prec5_m.item(), data.shape[0])
            top1_a.update(prec1_a.item(), data.shape[0])
            top5_a.update(prec5_a.item(), data.shape[0])

            # pdb.set_trace()
            batch_timer.update(time.time() - end)
            end = time.time()
            if results_m is not None:
                np.concatenate((results_m, output_m.cpu().numpy()), axis=0)
            else:
                results_m = output_m.cpu().numpy()

            if results_a is not None:
                np.concatenate((results_a, output_a.cpu().numpy()), axis=0)
            else:
                results_a = output_a.cpu().numpy()
            logging.info("{0}/{1} done, Batch: {batch_timer.val:.3f}({batch_timer.avg:.3f}), maxTop1: {top1_m.val:>6.3f}({top1_m.avg:>6.3f}), maxTop5: {top5_m.val:>6.3f}({top5_m.avg:>6.3f}), avgTop1: {top1_a.val:>6.3f}({top1_a.avg:>6.3f}), avgTop5: {top5_a.val:>6.3f}({top5_a.avg:>6.3f})".
                format(ind + 1, len(test_loader), 
                    batch_timer=batch_timer, 
                    top1_m=top1_m, top5_m=top5_m, top1_a=top1_a, top5_a=top5_a))
    max_target_file = os.path.join(args.experiment_root, "arch_{0}-epoch_{1}-top1_{2}-top5_{3}_max.npz".format(arch, test_epoch, top1_m.avg, top5_m.avg))
    avg_target_file = os.path.join(args.experiment_root, "arch_{0}-epoch_{1}-top1_{2}-top5_{3}_avg.npz".format(arch, test_epoch, top1_a.avg, top5_a.avg))
    print("saving {}".format(max_target_file))
    np.savez(max_target_file, results_m)
    print("saving {}".format(avg_target_file))
    np.savez(avg_target_file, results_a)
if __name__ == "__main__":
    main()
