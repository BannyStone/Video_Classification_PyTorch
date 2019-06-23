import os
from torch import nn
from torch.nn.parameter import Parameter
from .networks import *

from .transforms import *

class VideoModule(nn.Module):
    def __init__(self, num_class, base_model_name='resnet50', 
                 before_softmax=True, dropout=0.8, pretrained=True, pretrained_model=None):
        super(VideoModule, self).__init__()
        self.num_class = num_class
        self.base_model_name = base_model_name
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.pretrained = pretrained
        self.pretrained_model = pretrained_model
        # self.finetune = finetune

        self._prepare_base_model(base_model_name)

        if not self.before_softmax:
            self.softmax = nn.Softmax()

    def _prepare_base_model(self, base_model_name):
        """
        base_model+(dropout)+classifier
        """
        base_model_dict = None
        classifier_dict = None
        if self.pretrained and self.pretrained_model:
            model_dict = torch.load(self.pretrained_model)
            base_model_dict = {k: v for k, v in model_dict.items() if "classifier" not in k}
            classifier_dict = {'.'.join(k.split('.')[1:]): v for k, v in model_dict.items() if "classifier" in k}
        # base model
        if "resnet" in base_model_name:
            self.base_model = eval(base_model_name)(pretrained=self.pretrained, \
                                   feat=True, pretrained_model=base_model_dict)
        elif base_model_name == "mnet2":
            model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 
                                      "../models/mobilenet_v2.pth.tar")
            self.base_model = mnet2(pretrained=model_path, feat=True)
        elif base_model_name == "mnet2_3d":
            model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 
                                      "../models/mobilenet_v2.pth.tar")
            self.base_model = mnet2_3d(pretrained=model_path, feat=True)
        elif "fst" in base_model_name or "msv" in base_model_name or "gsv" in base_model_name:
            self.base_model = eval(base_model_name)(pretrained=self.pretrained,
                                    feat=True, pretrained_model=base_model_dict)
        else:
            raise ValueError('Unknown base model: {}'.format(base_model))

        # classifier: (dropout) + fc
        if self.dropout == 0:
            self.classifier = nn.Linear(self.base_model.feat_dim, self.num_class)
        elif self.dropout > 0:
            self.classifier = nn.Sequential(nn.Dropout(self.dropout), nn.Linear(self.base_model.feat_dim, self.num_class))

        # init classifier
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='linear')
                nn.init.constant_(m.bias, 0)
        
        # if self.pretrained and self.pretrained_model:
        #     self.classifier.load_state_dict(classifier_dict)

        # if self.finetune:
        #     print("Finetune")
        #     for param in self.base_model.parameters():
        #         param.requires_grad = False
        #     for m in self.base_model.modules():
        #         if isinstance(m, nn.BatchNorm3d):
        #             m.eval()

        # import pdb
        # pdb.set_trace()

    def forward(self, input):
        out = self.base_model(input)
        out = self.classifier(out)

        if not self.before_softmax:
            out = self.softmax(out)

        return out

    def get_augmentation(self):
        return torchvision.transforms.Compose([GroupMultiScaleCrop(input_size=224, scales=[1, .875, .75, .66]),
                                                   GroupRandomHorizontalFlip()])

class TSN(nn.Module):
    """Temporal Segment Network
    
    """
    def __init__(self, batch_size, video_module, num_segments=1, t_length=1, 
                 crop_fusion_type='max', mode="3D"):
        super(TSN, self).__init__()
        self.t_length = t_length
        self.batch_size = batch_size
        self.num_segments = num_segments
        self.video_module = video_module
        self.crop_fusion_type = crop_fusion_type
        self.mode = mode

    def forward(self, input):
        # reshape input first
        shape = input.shape
        if "3D" in self.mode:
            assert(len(shape)) == 5, "In 3D mode, input must have 5 dims."
            shape = (shape[0], shape[1], shape[2]//self.t_length, self.t_length) + shape[3:]
            input = input.view(shape).permute((0, 2, 1, 3, 4, 5)).contiguous()
            shape = (input.shape[0] * input.shape[1], ) + input.shape[2:]
            input = input.view(shape)
        elif "2D" in self.mode:
            assert(len(shape)) == 4, "In 2D mode, input must have 4 dims."
            shape = (shape[0]*shape[1]//3, 3,) + shape[2:]
            input = input.view(shape)
        else:
            raise Exception("Unsupported mode.")

        # base network forward
        output = self.video_module(input)
        # fuse output
        output = output.view((self.batch_size, 
                              output.shape[0] // (self.batch_size * self.num_segments), 
                              self.num_segments, output.shape[1]))

        output_max = output.max(1)[0].squeeze(1)
        pred_max = output_max.mean(1).squeeze(1)
        output_ave = output.mean(1).squeeze(1)
        pred_ave = output_ave.mean(1).squeeze(1)
        # if self.crop_fusion_type == 'max':
        #     # pdb.set_trace()
        #     output = output.max(1)[0].squeeze(1)
        # elif self.crop_fusion_type == 'avg':
        #     output = output.mean(1).squeeze(1)
        # pred = output.mean(1).squeeze(1)
        return (output_max, pred_max, output_ave, pred_ave)
