"""
Modify the original file to make the class support feature extraction
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.model_zoo as model_zoo

__all__ = ["resnet50_3d_p2stream"]

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class GloAvgPool3d(nn.Module):
    def __init__(self):
        super(GloAvgPool3d, self).__init__()
        self.stride = 1
        self.padding = 0
        self.ceil_mode = False
        self.count_include_pad = True

    def forward(self, input):
        input_shape = input.shape
        kernel_size = input_shape[2:]
        return F.avg_pool3d(input, kernel_size, self.stride,
                            self.padding, self.ceil_mode, self.count_include_pad)

class Bottleneck3D_000(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, t_stride=1, downsample=None):
        super(Bottleneck3D_000, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, 
                               stride=[t_stride, 1, 1], bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(1, 3, 3), 
                               stride=[1, stride, stride], padding=(0, 1, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class P2streamBlock(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, t_stride=1, downsample=None):
        super(P2streamBlock, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, 
                               stride=1,
                               padding=0, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        # self.bn1_t = nn.BatchNorm3d(planes)
        
        self.conv2_s = nn.Conv3d(planes, planes//2, kernel_size=(1, 3, 3), 
                               stride=(1, stride, stride), padding=(0, 1, 1), bias=False)
        self.conv2_t = nn.Conv3d(planes, planes//2, kernel_size=(3, 1, 1),
                                stride=(1, 1, 1), padding=(1, 0, 0), bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        # self.bn2_t = nn.BatchNorm3d(planes/2)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x # [N,C,T,H,W]

        xs = F.avg_pool3d(x, kernel_size=(3,1,1), stride=(2,1,1), padding=(1,0,0)) #[N,C,T//2,H,W]
        if x.shape[3] % 2 == 0:
            xt = F.avg_pool3d(x, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)) #[N,C,T,H//2,W//2]
        else:
            xt = x

        # Spatial Stream Conv1
        out_s = self.conv1(xs)
        out_s = self.bn1(out_s)
        out_s = self.relu(out_s) #[N,C//4,T//2,H,W]

        # Temporal Stream Conv1
        out_t = self.conv1(xt)
        out_t = self.bn1(out_t)
        out_t = self.relu(out_t) #[N,C//4,T,H//2,W//2]

        # Spatial Stream Conv2
        out_s = self.conv2_s(out_s)
        # out_s = self.bn2_s(out_s)
        # out_s = self.relu(out_s) #[N,C//8,T//2,H,W]

        # Temporal Stream Conv2
        out_t = self.conv2_t(out_t)
        # out_t = self.bn2_t(out_t)
        # out_t = self.relu(out_t) #[N,C//8,T,H//2,W//2]

        # Fusion
        out_s = F.interpolate(out_s, scale_factor=(2,1,1), mode='trilinear', align_corners=True) #[N,C//8,T,H,W]
        if self.stride == 1 and x.shape[3] % 2 == 0:
            out_t = F.interpolate(out_t, scale_factor=(1,2,2), mode='trilinear', align_corners=True) #[N,C//8,T,H,W]
        elif self.stride == 2 or x.shape[3] % 2 != 0:
            pass
        else:
            raise ValueError("Stride Value must be 1 or 2")

        # import pdb
        # pdb.set_trace()

        out = torch.cat((out_s, out_t), dim=1) #[N,C//4,T,H,W]

        out = self.bn2(out)
        out = self.relu(out) #[N,C//4,T,H,W]

        out = self.conv3(out)
        out = self.bn3(out) #[N,C,T,H,W]

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet3D_nodown(nn.Module):

    def __init__(self, block, layers, num_classes=1000, feat=False, **kwargs):
        if not isinstance(block, list):
            block = [block] * 4
        else:
            assert(len(block)) == 4, "Block number must be 4 for ResNet-Stype networks."
        self.inplanes = 64
        super(ResNet3D_nodown, self).__init__()
        self.feat = feat
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(1, 7, 7), 
                               stride=(1, 2, 2), padding=(0, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.layer1 = self._make_layer(block[0], 64, layers[0])
        self.layer2 = self._make_layer(block[1], 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block[2], 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block[3], 512, layers[3], stride=2)
        self.avgpool = GloAvgPool3d()
        self.feat_dim = 512 * block[0].expansion
        if not feat:
            self.fc = nn.Linear(512 * block[0].expansion, num_classes)

        for n, m in self.named_modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, t_stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=(t_stride, stride, stride), bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, t_stride=t_stride, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        if not self.feat:
            print("WARNING!!!!!!!")
            x = self.fc(x)

        return x

def part_state_dict(state_dict, model_dict):
    added_dict = {}
    for k, v in state_dict.items():
        if ".conv2.weight" in k and "layer" in k:
            in_channels = v.shape[1]
            out_channels = v.shape[0]
            new_k = k[:k.index(".conv2.weight")]+'.conv2_s.weight'
            added_dict.update({new_k: v[:out_channels//2,...]})
            new_k = k[:k.index(".conv2.weight")]+'.conv2_t.weight'
            # import pdb
            # pdb.set_trace()
            added_dict.update({new_k: v[out_channels//2:,...].sum(dim=2,keepdim=True).sum(dim=3,keepdim=True)})
        # elif ".bn2.weight" in k and "layer" in k:
        #     channels = v.shape[0]
        #     new_k = k[:k.index(".bn2.weight")]+'.bn2_s.weight'
        #     added_dict.update({new_k: v[:out_channels//2]})
        #     new_k = k[:k.index(".bn2.weight")]+'.bn2_t.weight'
        #     import pdb
        #     pdb.set_trace()
        #     added_dict.update({new_k: v[out_channels//2:]})
        # elif ".bn2.bias" in k and "layer" in k:
        #     channels = v.shape[0]
        #     new_k = k[:k.index(".bn2.bias")]+'.bn2_s.bias'
        #     added_dict.update({new_k: v[:out_channels//2]})
        #     new_k = k[:k.index(".bn2.bias")]+'.bn2_t.bias'
        #     import pdb
        #     pdb.set_trace()
        #     added_dict.update({new_k: v[out_channels//2:]})
        # elif ".bn2.running_mean" in k and "layer" in k:
        #     channels = v.shape[0]
        #     new_k = k[:k.index(".bn2.running_mean")]+'.bn2_s.running_mean'
        #     added_dict.update({new_k: v[:out_channels//2]})
        #     new_k = k[:k.index(".bn2.running_mean")]+'.bn2_t.running_mean'
        #     import pdb
        #     pdb.set_trace()
        #     added_dict.update({new_k: v[out_channels//2:]})
        # elif ".bn2.running_var" in k and "layer" in k:
        #     channels = v.shape[0]
        #     new_k = k[:k.index(".bn2.running_var")]+'.bn2_s.running_var'
        #     added_dict.update({new_k: v[:out_channels//2]})
        #     new_k = k[:k.index(".bn2.running_var")]+'.bn2_t.running_var'
        #     import pdb
        #     pdb.set_trace()
        #     added_dict.update({new_k: v[out_channels//2:]})
        # elif ".bn2.num_batches_tracked" in k and "layer" in k:
        #     channels = v.shape[0]
        #     new_k = k[:k.index(".bn2.num_batches_tracked")]+'.bn2_s.num_batches_tracked'
        #     added_dict.update({new_k: v[:out_channels//2]})
        #     new_k = k[:k.index(".bn2.num_batches_tracked")]+'.bn2_t.num_batches_tracked'
        #     import pdb
        #     pdb.set_trace()
        #     added_dict.update({new_k: v[out_channels//2:]})

    state_dict.update(added_dict)
    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    pretrained_dict = inflate_state_dict(pretrained_dict, model_dict)
    model_dict.update(pretrained_dict)
    return model_dict

def inflate_state_dict(pretrained_dict, model_dict):
    for k in pretrained_dict.keys():
        if pretrained_dict[k].size() != model_dict[k].size():
            assert(pretrained_dict[k].size()[:2] == model_dict[k].size()[:2]), \
                   "To inflate, channel number should match."
            assert(pretrained_dict[k].size()[-2:] == model_dict[k].size()[-2:]), \
                   "To inflate, spatial kernel size should match."
            print("Layer {} needs inflation.".format(k))
            shape = list(pretrained_dict[k].shape)
            shape.insert(2, 1)
            t_length = model_dict[k].shape[2]
            pretrained_dict[k] = pretrained_dict[k].reshape(shape)
            if t_length != 1:
                pretrained_dict[k] = pretrained_dict[k].expand_as(model_dict[k]) / t_length
            assert(pretrained_dict[k].size() == model_dict[k].size()), \
                   "After inflation, model shape should match."

    return pretrained_dict

def resnet50_3d_p2stream(pretrained=False, feat=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet3D_nodown([Bottleneck3D_000, Bottleneck3D_000, P2streamBlock, P2streamBlock], 
                     [3, 4, 6, 3], feat=feat, **kwargs)
    # import pdb
    # pdb.set_trace()
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
