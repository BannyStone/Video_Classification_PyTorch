"""
Modify the original file to make the class support feature extraction
"""
import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import math

from ..modules.fst import prenormFST as FST
from operator import mul
from functools import reduce

class basicFST(nn.Module):
    def __init__(self,
                inplanes, planes, 
                s_kernel_size=3, t_kernel_size=3, 
                s_stride=1, t_stride=1, 
                s_dilation=1, t_dilation=1, 
                wide=True, reduction_ratio=1):
        super(basicFST, self).__init__()
        if wide:
            c3d_params = inplanes * planes * s_kernel_size * s_kernel_size * t_kernel_size
            mid_planes = math.floor(c3d_params / (s_kernel_size * s_kernel_size * inplanes + t_kernel_size * planes))
        else:
            mid_planes = inplanes

        mid_planes //= reduction_ratio

        self.fst = FST(inplanes, mid_planes, planes, 
                        s_kernel_size=s_kernel_size,
                        t_kernel_size=t_kernel_size,
                        s_stride=s_stride,
                        t_stride=t_stride,
                        s_dilation=s_dilation,
                        t_dilation=t_dilation)
    
    def forward(self, x):

        x = self.fst(x)

        return x

class biggerFST(nn.Module):
    def __init__(self,
                inplanes, planes, 
                s_kernel_size=3, t_kernel_size=3, 
                s_stride=1, t_stride=1, 
                s_dilation=1, t_dilation=1, 
                wide=True, reduction_ratio=1):
        super(biggerFST, self).__init__()
        if wide:
            c3d_params = inplanes * planes * 3 * 3 * 3
            mid_planes = math.floor(c3d_params / (3 * 3 * inplanes + 3 * planes))
        else:
            mid_planes = inplanes

        mid_planes //= reduction_ratio

        self.fst = FST(inplanes, mid_planes, planes, 
                        s_kernel_size=s_kernel_size,
                        t_kernel_size=t_kernel_size,
                        s_stride=s_stride,
                        t_stride=t_stride,
                        s_dilation=s_dilation,
                        t_dilation=t_dilation)
    
    def forward(self, x):

        x = self.fst(x)

        return x

class FSTBlock(nn.Module):
    def __init__(self, inplanes, planes, s_stride=1, t_stride=1, downsample=None):
        super(FSTBlock, self).__init__()
        self.conv1 = basicFST(inplanes, planes, 
                        s_kernel_size=3, t_kernel_size=3, 
                        s_stride=s_stride, t_stride=t_stride)
        self.conv2 = basicFST(planes, planes, 
                        s_kernel_size=3, t_kernel_size=3)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return out

class x2FSTBlock(nn.Module):
    def __init__(self, inplanes, planes, s_stride=1, t_stride=1, downsample=None):
        super(x2FSTBlock, self).__init__()
        self.conv1 = basicFST(inplanes, planes, 
                        s_kernel_size=3, t_kernel_size=3, 
                        s_stride=s_stride, t_stride=t_stride,
                        reduction_ratio=2)
        self.conv2 = basicFST(planes, planes, 
                        s_kernel_size=3, t_kernel_size=3,
                        reduction_ratio=2)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return out

class x3FSTBlock(nn.Module):
    def __init__(self, inplanes, planes, s_stride=1, t_stride=1, downsample=None):
        super(x3FSTBlock, self).__init__()
        self.conv1 = basicFST(inplanes, planes, 
                        s_kernel_size=3, t_kernel_size=3, 
                        s_stride=s_stride, t_stride=t_stride,
                        reduction_ratio=3)
        self.conv2 = basicFST(planes, planes, 
                        s_kernel_size=3, t_kernel_size=3,
                        reduction_ratio=3)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return out

class x4FSTBlock(nn.Module):
    def __init__(self, inplanes, planes, s_stride=1, t_stride=1, downsample=None):
        super(x4FSTBlock, self).__init__()
        self.conv1 = basicFST(inplanes, planes, 
                        s_kernel_size=3, t_kernel_size=3, 
                        s_stride=s_stride, t_stride=t_stride,
                        reduction_ratio=4)
        self.conv2 = basicFST(planes, planes, 
                        s_kernel_size=3, t_kernel_size=3,
                        reduction_ratio=4)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return out

class sd2FSTBlock(nn.Module):
    def __init__(self, inplanes, planes, s_stride=1, t_stride=1, downsample=None):
        super(sd2FSTBlock, self).__init__()
        self.conv1 = basicFST(inplanes, planes, 
                        s_kernel_size=3, t_kernel_size=3, 
                        s_stride=s_stride, t_stride=t_stride,
                        s_dilation=2)
        self.conv2 = basicFST(planes, planes, 
                        s_kernel_size=3, t_kernel_size=3,
                        s_dilation=2)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return out

class x4sd2FSTBlock(nn.Module):
    def __init__(self, inplanes, planes, s_stride=1, t_stride=1, downsample=None):
        super(x4sd2FSTBlock, self).__init__()
        self.conv1 = basicFST(inplanes, planes, 
                        s_kernel_size=3, t_kernel_size=3, 
                        s_stride=s_stride, t_stride=t_stride,
                        s_dilation=2, reduction_ratio=4)
        self.conv2 = basicFST(planes, planes, 
                        s_kernel_size=3, t_kernel_size=3,
                        s_dilation=2, reduction_ratio=4)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return out

class x4sf5FSTBlock(nn.Module):
    def __init__(self, inplanes, planes, s_stride=1, t_stride=1, downsample=None):
        super(x4sf5FSTBlock, self).__init__()
        self.conv1 = biggerFST(inplanes, planes, 
                        s_kernel_size=5, t_kernel_size=3, 
                        s_stride=s_stride, t_stride=t_stride,
                        s_dilation=1, reduction_ratio=4)
        self.conv2 = biggerFST(planes, planes, 
                        s_kernel_size=5, t_kernel_size=3,
                        s_dilation=1, reduction_ratio=4)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return out

class x4sd4FSTBlock(nn.Module):
    def __init__(self, inplanes, planes, s_stride=1, t_stride=1, downsample=None):
        super(x4sd4FSTBlock, self).__init__()
        self.conv1 = basicFST(inplanes, planes, 
                        s_kernel_size=3, t_kernel_size=3, 
                        s_stride=s_stride, t_stride=t_stride,
                        s_dilation=4, reduction_ratio=4)
        self.conv2 = basicFST(planes, planes, 
                        s_kernel_size=3, t_kernel_size=3,
                        s_dilation=4, reduction_ratio=4)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return out

class sd4FSTBlock(nn.Module):
    def __init__(self, inplanes, planes, s_stride=1, t_stride=1, downsample=None):
        super(sd4FSTBlock, self).__init__()
        self.conv1 = basicFST(inplanes, planes, 
                        s_kernel_size=3, t_kernel_size=3, 
                        s_stride=s_stride, t_stride=t_stride,
                        s_dilation=4)
        self.conv2 = basicFST(planes, planes, 
                        s_kernel_size=3, t_kernel_size=3,
                        s_dilation=4)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return out

class sd8FSTBlock(nn.Module):
    def __init__(self, inplanes, planes, s_stride=1, t_stride=1, downsample=None):
        super(sd8FSTBlock, self).__init__()
        self.conv1 = basicFST(inplanes, planes, 
                        s_kernel_size=3, t_kernel_size=3, 
                        s_stride=s_stride, t_stride=t_stride,
                        s_dilation=8)
        self.conv2 = basicFST(planes, planes, 
                        s_kernel_size=3, t_kernel_size=3,
                        s_dilation=8)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return out

class FSTNet(nn.Module):

    def __init__(self, block_list, layers, num_classes=1000, feat=False, **kwargs):
        self.inplanes = 64
        super(FSTNet, self).__init__()
        self.feat = feat
        self.conv1 = nn.Conv3d(3, self.inplanes, kernel_size=(1, 7, 7), 
                                padding=(0, 3, 3), stride=(1, 2, 2))
        self.layer1 = self._make_layer(block_list[0], 64,  layers[0])
        self.layer2 = self._make_layer(block_list[1], 128, layers[1], s_stride=2, t_stride=2)
        self.layer3 = self._make_layer(block_list[2], 256, layers[2], s_stride=2, t_stride=2)
        self.layer4 = self._make_layer(block_list[3], 512, layers[3], s_stride=2, t_stride=2)
        self.avgpool = nn.AvgPool3d(kernel_size=(2, 7, 7), stride=1)
        self.feat_dim = 512
        if not feat:
            self.fc = nn.Linear(512, num_classes)

        for n, m in self.named_modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, number, s_stride=1, t_stride=1):
        downsample = None
        if s_stride != 1 or t_stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(nn.BatchNorm3d(self.inplanes), 
                                    nn.ReLU(inplace=True), 
                                    nn.Conv3d(self.inplanes, planes, 
                                    kernel_size=(3,1,1), 
                                    stride=(t_stride,s_stride,s_stride),
                                    padding=(1,0,0)))

        layers = []
        layers.append(block(self.inplanes, planes, s_stride=s_stride, t_stride=t_stride, downsample=downsample))
        self.inplanes = planes
        for i in range(1, number):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if not self.feat:
            x = self.fc(x)

        return x

def fst_resnet18(pretrained=False, feat=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = FSTNet((FSTBlock,) * 4, [2, 2, 2, 2], feat=feat, **kwargs)
    if pretrained:
        if kwargs['pretrained_model'] is None:
            state_dict = model_zoo.load_url(model_urls['resnet50'])
        else:
            print("Using specified pretrain model")
            state_dict = kwargs['pretrained_model']
        if feat:
            new_state_dict = part_state_dict(state_dict, model.state_dict())
            model.load_state_dict(new_state_dict)
    return model

def fst_resnet18_x2(pretrained=False, feat=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = FSTNet((x2FSTBlock,) * 4, [2, 2, 2, 2], feat=feat, **kwargs)
    if pretrained:
        if kwargs['pretrained_model'] is None:
            state_dict = model_zoo.load_url(model_urls['resnet50'])
        else:
            print("Using specified pretrain model")
            state_dict = kwargs['pretrained_model']
        if feat:
            new_state_dict = part_state_dict(state_dict, model.state_dict())
            model.load_state_dict(new_state_dict)
    return model

def fst_resnet18_x3(pretrained=False, feat=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = FSTNet((x3FSTBlock,) * 4, [2, 2, 2, 2], feat=feat, **kwargs)
    if pretrained:
        if kwargs['pretrained_model'] is None:
            state_dict = model_zoo.load_url(model_urls['resnet50'])
        else:
            print("Using specified pretrain model")
            state_dict = kwargs['pretrained_model']
        if feat:
            new_state_dict = part_state_dict(state_dict, model.state_dict())
            model.load_state_dict(new_state_dict)
    return model

def fst_resnet18_x4(pretrained=False, feat=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = FSTNet((x4FSTBlock,) * 4, [2, 2, 2, 2], feat=feat, **kwargs)
    if pretrained:
        if kwargs['pretrained_model'] is None:
            state_dict = model_zoo.load_url(model_urls['resnet50'])
        else:
            print("Using specified pretrain model")
            state_dict = kwargs['pretrained_model']
        if feat:
            new_state_dict = part_state_dict(state_dict, model.state_dict())
            model.load_state_dict(new_state_dict)
    return model

def fst_resnet18_sd2_st1(pretrained=False, feat=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = FSTNet((sd2FSTBlock,FSTBlock,FSTBlock,FSTBlock), [2, 2, 2, 2], feat=feat, **kwargs)
    if pretrained:
        if kwargs['pretrained_model'] is None:
            state_dict = model_zoo.load_url(model_urls['resnet50'])
        else:
            print("Using specified pretrain model")
            state_dict = kwargs['pretrained_model']
        if feat:
            new_state_dict = part_state_dict(state_dict, model.state_dict())
            model.load_state_dict(new_state_dict)
    return model

def fst_resnet18_sd2_st1_x4(pretrained=False, feat=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = FSTNet((x4sd2FSTBlock,x4FSTBlock,x4FSTBlock,x4FSTBlock), [2, 2, 2, 2], feat=feat, **kwargs)
    if pretrained:
        if kwargs['pretrained_model'] is None:
            state_dict = model_zoo.load_url(model_urls['resnet50'])
        else:
            print("Using specified pretrain model")
            state_dict = kwargs['pretrained_model']
        if feat:
            new_state_dict = part_state_dict(state_dict, model.state_dict())
            model.load_state_dict(new_state_dict)
    return model

def fst_resnet18_sf5_st1_x4(pretrained=False, feat=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = FSTNet((x4sf5FSTBlock,x4FSTBlock,x4FSTBlock,x4FSTBlock), [2, 2, 2, 2], feat=feat, **kwargs)
    if pretrained:
        if kwargs['pretrained_model'] is None:
            state_dict = model_zoo.load_url(model_urls['resnet50'])
        else:
            print("Using specified pretrain model")
            state_dict = kwargs['pretrained_model']
        if feat:
            new_state_dict = part_state_dict(state_dict, model.state_dict())
            model.load_state_dict(new_state_dict)
    return model

def fst_resnet18_sd4_st1_x4(pretrained=False, feat=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = FSTNet((x4sd4FSTBlock,x4FSTBlock,x4FSTBlock,x4FSTBlock), [2, 2, 2, 2], feat=feat, **kwargs)
    if pretrained:
        if kwargs['pretrained_model'] is None:
            state_dict = model_zoo.load_url(model_urls['resnet50'])
        else:
            print("Using specified pretrain model")
            state_dict = kwargs['pretrained_model']
        if feat:
            new_state_dict = part_state_dict(state_dict, model.state_dict())
            model.load_state_dict(new_state_dict)
    return model

def fst_resnet18_sd4_st4_x4(pretrained=False, feat=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = FSTNet((x4FSTBlock,x4FSTBlock,x4FSTBlock,x4sd4FSTBlock), [2, 2, 2, 2], feat=feat, **kwargs)
    if pretrained:
        if kwargs['pretrained_model'] is None:
            state_dict = model_zoo.load_url(model_urls['resnet50'])
        else:
            print("Using specified pretrain model")
            state_dict = kwargs['pretrained_model']
        if feat:
            new_state_dict = part_state_dict(state_dict, model.state_dict())
            model.load_state_dict(new_state_dict)
    return model

def fst_resnet18_sd2_st4_x4(pretrained=False, feat=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = FSTNet((x4FSTBlock,x4FSTBlock,x4FSTBlock,x4sd2FSTBlock), [2, 2, 2, 2], feat=feat, **kwargs)
    if pretrained:
        if kwargs['pretrained_model'] is None:
            state_dict = model_zoo.load_url(model_urls['resnet50'])
        else:
            print("Using specified pretrain model")
            state_dict = kwargs['pretrained_model']
        if feat:
            new_state_dict = part_state_dict(state_dict, model.state_dict())
            model.load_state_dict(new_state_dict)
    return model

def fst_resnet18_sd4_st1(pretrained=False, feat=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = FSTNet((sd4FSTBlock,FSTBlock,FSTBlock,FSTBlock), [2, 2, 2, 2], feat=feat, **kwargs)
    if pretrained:
        if kwargs['pretrained_model'] is None:
            state_dict = model_zoo.load_url(model_urls['resnet50'])
        else:
            print("Using specified pretrain model")
            state_dict = kwargs['pretrained_model']
        if feat:
            new_state_dict = part_state_dict(state_dict, model.state_dict())
            model.load_state_dict(new_state_dict)
    return model

def fst_resnet18_sd8_st1(pretrained=False, feat=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = FSTNet((sd8FSTBlock,FSTBlock,FSTBlock,FSTBlock), [2, 2, 2, 2], feat=feat, **kwargs)
    if pretrained:
        if kwargs['pretrained_model'] is None:
            state_dict = model_zoo.load_url(model_urls['resnet50'])
        else:
            print("Using specified pretrain model")
            state_dict = kwargs['pretrained_model']
        if feat:
            new_state_dict = part_state_dict(state_dict, model.state_dict())
            model.load_state_dict(new_state_dict)
    return model

def fst_resnet18_sd2_st2(pretrained=False, feat=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = FSTNet((FSTBlock,sd2FSTBlock,FSTBlock,FSTBlock), [2, 2, 2, 2], feat=feat, **kwargs)
    if pretrained:
        if kwargs['pretrained_model'] is None:
            state_dict = model_zoo.load_url(model_urls['resnet50'])
        else:
            print("Using specified pretrain model")
            state_dict = kwargs['pretrained_model']
        if feat:
            new_state_dict = part_state_dict(state_dict, model.state_dict())
            model.load_state_dict(new_state_dict)
    return model

def fst_resnet18_sd2_st3(pretrained=False, feat=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = FSTNet((FSTBlock,FSTBlock,sd2FSTBlock,FSTBlock), [2, 2, 2, 2], feat=feat, **kwargs)
    if pretrained:
        if kwargs['pretrained_model'] is None:
            state_dict = model_zoo.load_url(model_urls['resnet50'])
        else:
            print("Using specified pretrain model")
            state_dict = kwargs['pretrained_model']
        if feat:
            new_state_dict = part_state_dict(state_dict, model.state_dict())
            model.load_state_dict(new_state_dict)
    return model

def fst_resnet18_sd2_st4(pretrained=False, feat=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = FSTNet((FSTBlock,FSTBlock,FSTBlock,sd2FSTBlock), [2, 2, 2, 2], feat=feat, **kwargs)
    if pretrained:
        if kwargs['pretrained_model'] is None:
            state_dict = model_zoo.load_url(model_urls['resnet50'])
        else:
            print("Using specified pretrain model")
            state_dict = kwargs['pretrained_model']
        if feat:
            new_state_dict = part_state_dict(state_dict, model.state_dict())
            model.load_state_dict(new_state_dict)
    return model