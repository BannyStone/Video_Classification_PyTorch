import torch
import torch.nn as nn
import math
import os

__all__ = ["mnet2_3d"]

def conv_bn(inp, oup, stride, t_stride=1):
    return nn.Sequential(
        nn.Conv3d(inp, oup, kernel_size=(1, 3, 3), 
                  stride=(t_stride, stride, stride), padding=(0, 1, 1), bias=False),
        nn.BatchNorm3d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv3d(inp, oup, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm3d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, t_stride, expand_ratio, t_radius=1):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.t_stride = t_stride
        self.t_radius = t_radius
        assert stride in [1, 2] and t_stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            assert(t_stride == 1), "Temporal stride must be one when expand ratio is one."
            self.conv = nn.Sequential(
                # dw
                nn.Conv3d(hidden_dim, hidden_dim, kernel_size=(1, 3, 3), stride=(t_stride, stride, stride), 
                          padding=(0, 1, 1), groups=hidden_dim, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv3d(hidden_dim, oup, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm3d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv3d(inp, hidden_dim, kernel_size=(t_radius * 2 + 1, 1, 1), 
                          stride=(t_stride, 1, 1), padding=(t_radius, 0, 0), bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv3d(hidden_dim, hidden_dim, kernel_size=(1, 3, 3), stride=(1, stride, stride), 
                          padding=(0, 1, 1), groups=hidden_dim, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv3d(hidden_dim, oup, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm3d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2_3D(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1., feat=False):
        super(MobileNetV2_3D, self).__init__()
        self.feat = feat
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s, ts, r
            [1, 16, 1, 1, 1, 0],
            [6, 24, 2, 2, 1, 0],
            [6, 32, 3, 2, 1, 0],
            [6, 64, 4, 2, 1, 1],
            [6, 96, 3, 1, 2, 1],
            [6, 160, 3, 2, 2, 1],
            [6, 320, 1, 1, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.feat_dim = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s, ts, r in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, ts, expand_ratio=t, t_radius=r))
                else:
                    self.features.append(block(input_channel, output_channel, 1, 1, expand_ratio=t, t_radius=r))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1x1_bn(input_channel, self.feat_dim))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)
        self.avgpool = nn.AvgPool3d(kernel_size=(4, 7, 7), stride=1)

        # building classifier
        if not self.feat:
            self.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(self.feat_dim, n_class),
                )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if not self.feat:
            x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def part_state_dict(state_dict, model_dict):
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

def mnet2_3d(pretrained=None, feat=False):
    if pretrained != None:
        assert(os.path.exists(pretrained)), "pretrained model does not exist."
    model = MobileNetV2_3D(feat=feat)
    if pretrained:
        state_dict = torch.load(pretrained, map_location=lambda storage, loc: storage)
        state_dict = part_state_dict(state_dict, model.state_dict())
        model.load_state_dict(state_dict)
    return model
