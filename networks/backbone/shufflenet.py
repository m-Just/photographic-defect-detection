from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from networks.building_blocks import conv3x3, conv1x1, channel_shuffle

__all__ = ['ShuffleNet', 'shufflenet']


class ShuffleUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=3,
                 grouped_conv=True, combine='add'):
        super(ShuffleUnit, self).__init__()

        self.featmaps = []

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grouped_conv = grouped_conv
        self.combine = combine
        self.groups = groups
        self.bottleneck_channels = self.out_channels // 4

        # define the type of ShuffleUnit
        if self.combine == 'add':
            # ShuffleUnit Figure 2b
            self.depthwise_stride = 1
            self._combine_func = self._add
        elif self.combine == 'concat':
            # ShuffleUnit Figure 2c
            self.depthwise_stride = 2
            self._combine_func = self._concat

            # ensure output of concat has the same channels as
            # original output channels.
            self.out_channels -= self.in_channels
        else:
            raise ValueError("Cannot combine tensors with \"{}\"" \
                             "Only \"add\" and \"concat\" are" \
                             "supported".format(self.combine))

        # Use a 1x1 grouped or non-grouped convolution to reduce input channels
        # to bottleneck channels, as in a ResNet bottleneck module.
        # NOTE: Do not use group convolution for the first conv1x1 in Stage 2.
        self.first_1x1_groups = self.groups if grouped_conv else 1

        self.g_conv_1x1_compress = self._make_grouped_conv1x1(
            self.in_channels,
            self.bottleneck_channels,
            self.first_1x1_groups,
            batch_norm=True,
            relu=True)

        # 3x3 depthwise convolution followed by batch normalization
        self.depthwise_conv3x3 = conv3x3(
            self.bottleneck_channels, self.bottleneck_channels,
            stride=self.depthwise_stride, groups=self.bottleneck_channels)
        self.bn_after_depthwise = nn.BatchNorm2d(self.bottleneck_channels)

        # Use 1x1 grouped convolution to expand from
        # bottleneck_channels to out_channels
        self.g_conv_1x1_expand = self._make_grouped_conv1x1(
            self.bottleneck_channels,
            self.out_channels,
            self.groups,
            batch_norm=True,
            relu=False)


    @staticmethod
    def _add(x, out):
        # residual connection
        return x + out


    @staticmethod
    def _concat(x, out):
        # concatenate along channel axis
        return torch.cat((x, out), 1)


    def _make_grouped_conv1x1(self, in_channels, out_channels, groups,
        batch_norm=True, relu=False):
        modules = OrderedDict()

        conv = conv1x1(in_channels, out_channels, groups=groups)
        modules['conv1x1'] = conv

        if batch_norm:
            modules['batch_norm'] = nn.BatchNorm2d(out_channels)
        if relu:
            modules['relu'] = nn.ReLU()
        if len(modules) > 1:
            return nn.Sequential(modules)
        else:
            return conv


    def forward(self, x):
        # save for combining later with output
        residual = x

        if self.combine == 'concat':
            residual = F.avg_pool2d(residual, kernel_size=3,
                stride=2, padding=1)

        out = self.g_conv_1x1_compress(x)
        out = channel_shuffle(out, self.groups)
        out = self.depthwise_conv3x3(out)
        out = self.bn_after_depthwise(out)
        out = self.g_conv_1x1_expand(out)
        out = self._combine_func(residual, out)
        out = F.relu(out)

        return out


class ShuffleNet(nn.Module):
    def __init__(self, groups=8, in_channels=3, trunc_stage=False,
                 reduce_channel=None):
        super(ShuffleNet, self).__init__()
        self.groups = groups
        self.trunc_stage = trunc_stage
        if self.trunc_stage:
            self.stage_repeats = [3, 3]
        else:
            self.stage_repeats = [3, 7, 3]

        # index 0 is invalid and should never be called.
        # only used for indexing convenience.
        if groups == 1:
            self.stage_out_channels = [-1, 24, 144, 288, 567]
        elif groups == 2:
            self.stage_out_channels = [-1, 24, 200, 400, 800]
        elif groups == 3:
            self.stage_out_channels = [-1, 24, 240, 480, 960]
        elif groups == 4:
            self.stage_out_channels = [-1, 24, 272, 544, 1088]
        elif groups == 8:
            self.stage_out_channels = [-1, 24, 384, 768, 1536]
        else:
            raise ValueError(
                """{} groups is not supported for
                   1x1 Grouped Convolutions""".format(groups))

        if reduce_channel is not None:
            raise NotImplementedError()

        # Stage 1 always has 24 output channels
        self.conv1 = conv3x3(in_channels,
                             self.stage_out_channels[1], # stage 1
                             stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Stage 2
        self.stage2 = self._make_stage(2)

        # Stage 3
        self.stage3 = self._make_stage(3)

        # Stage 4
        if self.trunc_stage:
            self.stage4 = None
            self.num_output_channels = self.stage_out_channels[3]
        else:
            self.stage4 = self._make_stage(4)
            self.num_output_channels = self.stage_out_channels[4]

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.01)
                # init.constant_(m.weight, 0.001)
                # print(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def _make_stage(self, stage):
        modules = OrderedDict()
        stage_name = f"ShuffleUnit_Stage{stage}"

        # First ShuffleUnit in the stage
        # 1. non-grouped 1x1 convolution (i.e. pointwise convolution)
        #   is used in Stage 2. Group convolutions used everywhere else.
        grouped_conv = stage > 2

        # 2. concatenation unit is always used.
        first_module = ShuffleUnit(
            self.stage_out_channels[stage-1],
            self.stage_out_channels[stage],
            groups=self.groups,
            grouped_conv=grouped_conv,
            combine='concat')
        modules[stage_name+"_0"] = first_module

        # add more ShuffleUnits depending on pre-defined number of repeats
        for i in range(self.stage_repeats[stage-2]):
            name = stage_name + f"_{i+1}"
            module = ShuffleUnit(
                self.stage_out_channels[stage],
                self.stage_out_channels[stage],
                groups=self.groups,
                grouped_conv=True,
                combine='add')
            modules[name] = module

        return nn.Sequential(modules)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        if self.stage4 is not None:
            x = self.stage4(x)
        return x

    def forward_kd(self, x):
        outputs = []
        x = self.conv1(x)
        x = self.maxpool(x)
        outputs.append(x)
        st = 4 if self.trunc_stage else 5
        for i in range(2, st):
            x = getattr(self, f'stage{i}')(x)
            outputs.append(x)
        return outputs


def shufflenet(**kwargs):
    return ShuffleNet(**kwargs)
