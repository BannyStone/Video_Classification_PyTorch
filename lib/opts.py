import os
import logging
import argparse

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

parser = argparse.ArgumentParser(description="PyTorch implementation of Video Classification")
parser.add_argument('dataset', type=str)
parser.add_argument('train_list', type=str)
parser.add_argument('val_list', type=str)

# ========================= Model Configs ==========================
parser.add_argument('--arch', '-a', type=str, default="resnet18")
parser.add_argument('--shadow', action='store_true')
parser.add_argument('--dropout', '--do', default=0.2, type=float,
                    metavar='DO', help='dropout ratio (default: 0.2)')
parser.add_argument('--mode', type=str, default='3D', choices=['3D', 'TSN', '2D'])
parser.add_argument('--new_size', type=int, default=256)
parser.add_argument('--crop_size', type=int, default=224)
parser.add_argument('--t_length', type=int, default=32, help="time length")
parser.add_argument('--t_stride', type=int, default=2, help="time stride between frames")
parser.add_argument('--num_segments', type=int, default=1)
parser.add_argument('--pretrained', action='store_true')
parser.add_argument('--pretrained_model', type=str, default=None)

# ========================= Learning Configs ==========================
parser.add_argument('--epochs', default=60, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_steps', default=[40, 70, 70], type=float, nargs="+",
                    metavar='LRSteps', help='epochs to decay learning rate by 10')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')

# ========================= Monitor Configs ==========================
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 20)')
parser.add_argument('--eval-freq', '-ef', default=2, type=int,
                    metavar='N', help='evaluation frequency (default: 2)')

# ========================= Runtime Configs ==========================
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--output_root', type=str, default="./output")
parser.add_argument('--image_tmpl', type=str, default="image_{:06d}.jpg")

args = parser.parse_args()
if args.mode == "2D":
     args.t_length = 1

experiment_id = '_'.join(map(str, [args.dataset, args.arch, args.mode, 
           'length'+str(args.t_length), 'stride'+str(args.t_stride), 
           'dropout'+str(args.dropout)]))

if args.pretrained and args.pretrained_model:
    if "2d" in args.pretrained_model:
        experiment_id += '_2dpretrained'

if args.shadow:
    experiment_id += '_shadow'

args.experiment_root = os.path.join(args.output_root, experiment_id)
# init logger
set_logger()
logging.info(args)
if not os.path.exists(args.experiment_root):
     os.makedirs(args.experiment_root)
