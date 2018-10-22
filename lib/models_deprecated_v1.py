import os
from torch import nn
from .networks.mnet2 import mnet2
from .networks.mnet2_3d import mnet2_3d
from .networks.resnet import *
from .networks.resnet_3d import *
from .networks.shadownet import resnet50_shadow

from .transforms import *

import ipdb

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
        
        if self.pretrained and self.pretrained_model:
            pass
            # print("load classifier")
            # self.classifier.load_state_dict(classifier_dict)

    def forward(self, input):
        out = self.base_model(input)
        out = self.classifier(out)

        if not self.before_softmax:
            out = self.softmax(out)

        return out

    def get_augmentation(self):
        return torchvision.transforms.Compose([GroupMultiScaleCrop(input_size=224, scales=[1, .875, .75, .66]),
                                                   GroupRandomHorizontalFlip()])

class VideoShadowModule(nn.Module):
    def __init__(self, num_class, base_model_name='resnet50_3d', 
                 before_softmax=True, dropout=0.8, pretrained=True, pretrained_model=None):
        super(VideoShadowModule, self).__init__()
        self.num_class = num_class
        self.base_model_name = base_model_name
        self.before_softmax = before_softmax
        self.dropout = dropout
        self.pretrained = pretrained
        self.pretrained_model = pretrained_model

        self._prepare_base_model(base_model_name)
        shadow_model_name = base_model_name.split('_')[0] + '_shadow'
        self._prepare_shadow_model(shadow_model_name)

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
        
        if self.pretrained and self.pretrained_model:
            pass
            # print("load classifier")
            # self.classifier.load_state_dict(classifier_dict)

    def _prepare_shadow_model(self, shadow_model_name):

        # base model (currently only support resnet50_shadow)
        if "resnet" in shadow_model_name:
            self.shadow_model = eval(shadow_model_name)(feat=True)
        else:
            raise ValueError('Unknown shadow model: {}'.format())

    def _cast_shadow(self, input):
        shadow_modules_dict = dict(self.shadow_model.named_modules())
        shadow_module_names = shadow_modules_dict.keys()
        # cast parameters
        # for p in self.base_model.named_parameters():
        #     print("input device: {}".format(input.device), p[0])
        for param_base in self.base_model.named_parameters():
            name = param_base[0]
            param = param_base[1]
            _items = name.split('.')
            module_name = '.'.join(_items[:-1])
            param_name = _items[-1]
            assert(param_name in ('weight', 'bias')), "parameter type must be weight or bias"
            assert(module_name in shadow_module_names),"Name not in shadow_module_names"
            # casting
            shadow_module = shadow_modules_dict[module_name]
            if param.dim() == 5:
                param = param.sum(dim=2, keepdim=True)
            assert(param.shape == shadow_module.shapes[param_name]), "param shape mismatch"
            shadow_module.register_nonleaf_parameter(param_name, param)
            # if module_name == "conv1" and param_name == "weight":
            # assert(input.device == eval("shadow_module.{}.device".format(param_name))), "Error1--------------"
            # assert(input.device == eval("self.shadow_model.{}.{}.device".format(module_name, param_name))), "Error2---------"
                # print("Module_name: {} | Param_name: {} | Device: {} PASS-------------------".format(module_name, param_name, input.device))
                # break
        # print("input_device: {} | param_device: {}".format(input.device, self.shadow_model.conv1.weight.device))
        # print(input.device, id(self.shadow_model.conv1.weight))
        # assert(input.device == eval("self.shadow_model.{}.{}.device".format('conv1', 'weight'))), "Error2+---------"
        
            # print("Chekc:", eval("id(shadow_module.{})".format(param_name)) == id(param))
        # cast buffers
        # for buffer_base in self.base_model.named_buffers():
        #     name = buffer_base[0]
        #     buffer = buffer_base[1]
        #     _items = name.split('.')
        #     module_name = '.'.join(_items[:-1])
        #     buffer_name = _items[-1]
        #     assert(buffer_name in ('running_mean', 'running_var', 'num_batches_tracked')), "buffer type constrain"
        #     assert(module_name in shadow_module_names), "Name not in shadow_module_names"
        #     # casting
        #     shadow_module = shadow_modules_dict[module_name]
        #     # assert(buffer.shape == shadow_module.shapes[buffer_name]), "name :{} | {} shape mismatch: {}/{}".format(module_name, buffer_name, buffer.shape, shadow_module.shapes[buffer_name])
        #     shadow_module.register_buffer(buffer_name, buffer)

    def _aggregate(self, dense_pred, sparse_pred):
        assert(dense_pred.dim() == 2 and sparse_pred.dim() == 3), "Prediction dimension error."
        dense_pred = dense_pred.view(dense_pred.shape[0], dense_pred.shape[1], 1)
        out = torch.cat((dense_pred, sparse_pred), dim=2)
        num_segments = out.shape[2]
        out = out.sum(dim=2, keepdim=False).div(num_segments)
        return out

    def forward(self, input):
        # Infer 3D network
        out1 = self.base_model(input[:,:,:16,...])
        # Cast Shadow
        self._cast_shadow(input)
        # Infer TSN
        out2 = self.shadow_model(input[:,:,16:,...])
        # Copy buffers back
        # self._copy_buffers_to_stereo()
        # Aggregate across segments
        out = self._aggregate(out1, out2)
        out = self.classifier(out)
        if not self.before_softmax:
            out = self.softmax(out)
        # ipdb.set_trace()

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
        if self.crop_fusion_type == 'max':
            # pdb.set_trace()
            output = output.max(1)[0].squeeze(1)
        elif self.crop_fusion_type == 'avg':
            output = output.mean(1).squeeze(1)
        pred = output.mean(1).squeeze(1)
        return (output, pred)
