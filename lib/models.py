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
            raise ValueError('Unknown shadow proto model: {}'.format())

    def _cast_params(self):
        for param_stereo, param_shadow in zip(self.base_model.named_parameters(), self.shadow_model.named_parameters()):
            # print("<--casting {}--> device: {}".format(param_stereo[0], param_stereo[1].device))
            # if param_stereo[0] == "conv1.weight":
                # ipdb.set_trace()
                # pass
            # print(param_stereo[1].is_leaf)
            # print(param_stereo[1].requires_grad)
            # print(param_shadow[1].is_leaf)
            # print(param_shadow[1].requires_grad)
            # print(param_stereo[1].device)
            # print(param_shadow[1].device)
            print("Param Name {}".format(param_stereo[0]), \
                  "| stereo leaf/req_grad", '/'.join([str(param_stereo[1].is_leaf), str(param_stereo[1].requires_grad)]), \
                  "| shadow leaf/req_grad", '/'.join([str(param_shadow[1].is_leaf), str(param_shadow[1].requires_grad)]), \
                  "| param device", param_stereo[1].device, param_shadow[1].device)
            assert(param_stereo[0] == param_shadow[0]), "Name mismatch."
            stereo_shape = param_stereo[1].shape
            shadow_shape = param_shadow[1].shape
            # with same shape, just copy
            if stereo_shape == shadow_shape:
                try:
                    param_shadow[1].copy_(param_stereo[1])
                except Exception as e:
                    print("Error message: {}\n".format(e))
                    # param_shadow[1].copy_(param_stereo[1])
            elif param_stereo[1].dim() == param_shadow[1].dim() == 5:
                assert(stereo_shape[:2] == shadow_shape[:2] and stereo_shape[3:] == shadow_shape[3:]), \
                        "Channel number and spatial dimension must match each other."
                assert(shadow_shape[2] == 1), \
                        "Shadow net conv weight parameters time dimension must be 1."
                try:
                    param_shadow[1].copy_(param_stereo[1].sum(dim=2, keepdim=True))
                except Exception as e:
                    print("Error message: {}\n".format(e))
            else:
                raise Exception("Wrong param pair {0} (stereo: {1}; shadow: {2})".format(param_stereo[0], 
                    stereo_shape, shadow_shape))
            # print("<--casted {}--> device: {}".format(param_stereo[0], param_stereo[1].device))

    def _copy_buffers_to_shadow(self):
        # print("copying buffers to shadow.")
        for buffer_stereo, buffer_shadow in zip(self.base_model.named_buffers(), self.shadow_model.named_buffers()):
            assert(buffer_stereo[0] == buffer_shadow[0]), "Name mismatch."
            stereo_shape = buffer_stereo[1].shape
            shadow_shape = buffer_shadow[1].shape
            # with same shape, just copy
            assert(stereo_shape == shadow_shape), "buffer shape must be the same."
            buffer_shadow[1].copy_(buffer_stereo[1])

    def _copy_buffers_to_stereo(self):
        # print("copying buffers to stereo.")
        for buffer_stereo, buffer_shadow in zip(self.base_model.named_buffers(), self.shadow_model.named_buffers()):
            assert(buffer_stereo[0] == buffer_shadow[0]), "Name mismatch."
            stereo_shape = buffer_stereo[1].shape
            shadow_shape = buffer_shadow[1].shape
            # with same shape, just copy
            assert(stereo_shape == shadow_shape), "buffer shape must be the same."
            buffer_stereo[1].copy_(buffer_shadow[1])
            # buffer_shadow[1].copy_(buffer_stereo[1])

    def _aggregate(self, dense_pred, sparse_pred):
        assert(dense_pred.dim() == 2 and sparse_pred.dim() == 3), "Prediction dimension error."
        dense_pred = dense_pred.view(dense_pred.shape[0], dense_pred.shape[1], 1)
        out = torch.cat((dense_pred, sparse_pred), dim=2)
        num_segments = out.shape[2]
        out = out.sum(dim=2, keepdim=False).div(num_segments)
        return out

    def forward(self, input):
        # Infer 3D network
        # ipdb.set_trace()
        out1 = self.base_model(input[:,:,:16,...])
        # print("conv1 weight device", self.base_model.conv1.weight.device)
        # Cast Shadow
        # ipdb.set_trace()
        # torch.cuda.synchronize()
        # ipdb.set_trace()
        print("first casting...")
        self._cast_params()
        # ipdb.set_trace()
        self._copy_buffers_to_shadow()
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
