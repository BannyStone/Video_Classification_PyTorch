import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import math

from ..modules.fst import sharenormFST as FST

class msvFST(nn.Module):
    def __init__(self, inplanes, planes, 
                s_kernel_size=3, t_kernel_size=3, 
                s_stride=1, t_stride=1, 
                wide=True, channel_reduction_ratio=1):
        super(msvFST, self).__init__()
        # following R(2+1)D
        if wide:
            c3d_params = inplanes * planes * s_kernel_size * s_kernel_size * t_kernel_size
            mid_planes = math.floor(c3d_params / (s_kernel_size * s_kernel_size * inplanes + t_kernel_size * planes))
        else:
            mid_planes = inplanes
        
        # reduce channel
        mid_planes = math.ceil(mid_planes / channel_reduction_ratio)

        # bn
        self.bn = nn.BatchNorm3d(inplanes)
        # relu
        self.relu = nn.ReLU(inplace=True)
        # branch 1: s_dilation=1, t_dilation=1
        self.fst1 = FST(inplanes, mid_planes, planes, 
                        s_stride=s_stride,
                        t_stride=t_stride,
                        s_dilation=1,
                        t_dilation=1)
        # branch 2: s_dilation=2, t_dilation=2
        self.fst2 = FST(inplanes, mid_planes, planes,
                        s_stride=s_stride,
                        t_stride=t_stride,
                        s_dilation=2,
                        t_dilation=1)
        # branch 3: s_dilation=3, t_dilation=3
        self.fst3 = FST(inplanes, mid_planes, planes,
                        s_stride=s_stride,
                        t_stride=t_stride,
                        s_dilation=3,
                        t_dilation=1)

    def forward(self, x):

        x = self.bn(x)
        x = self.relu(x)

        out1 = self.fst1(x)
        out2 = self.fst2(x)
        out3 = self.fst3(x)

        out = out1 + out2 + out3

        return out

class MSVBlock(nn.Module):
    def __init__(self, inplanes, planes, s_stride=1, t_stride=1, downsample=None):
        super(MSVBlock, self).__init__()
        self.conv1 = msvFST(inplanes, planes, 
                            s_kernel_size=3, t_kernel_size=3, 
                            s_stride=s_stride, t_stride=t_stride)
        self.conv2 = msvFST(planes, planes, 
                            s_kernel_size=3, t_kernel_size=3)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        return out

class liteMSVBlock(nn.Module):
    def __init__(self, inplanes, planes, s_stride=1, t_stride=1, downsample=None):
        super(liteMSVBlock, self).__init__()
        self.conv1 = msvFST(inplanes, planes, 
                            s_kernel_size=3, t_kernel_size=3, 
                            s_stride=s_stride, t_stride=t_stride,
                            wide=False)
        self.conv2 = msvFST(planes, planes, 
                            s_kernel_size=3, t_kernel_size=3,
                            wide=False)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        return out

class MSVNet(nn.Module):

    def __init__(self, block_list, layers, num_classes=1000, feat=False, **kwargs):
        self.inplanes = 64
        super(MSVNet, self).__init__()
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

def msv_resnet18(pretrained=False, feat=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = MSVNet((MSVBlock,) * 4, [2, 2, 2, 2], feat=feat, **kwargs)
    if pretrained:
        if kwargs['pretrained_model'] is None:
            raise ValueError("For MSVNet, pretrain model must be specified")
        else:
            print("Using specified pretrain model")
            state_dict = kwargs['pretrained_model']
        if feat:
            new_state_dict = part_state_dict(state_dict, model.state_dict())
            model.load_state_dict(new_state_dict)
    return model