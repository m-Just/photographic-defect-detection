import torch
import torch.nn as nn


def conv3x3(in_channels, out_channels, stride=1,
            padding=1, bias=True, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups,
        dilation=dilation)


def conv1x1(in_channels, out_channels, bias=True, groups=1, stride=1):
    """1x1 convolution with padding
    - Normal pointwise convolution When groups == 1
    - Grouped pointwise convolution when groups > 1
    """
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        bias=bias,
        groups=groups,
        stride=stride)


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class DummyModule(nn.Module):
    def __init__(self):
        super(DummyModule, self).__init__()

    def forward(self, x):
        return x


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride):
        super(InvertedResidual, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride > 1) else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out


class GroupedConvBlock(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_channels_per_group,
                 norm_type='batch_norm'):
        super(GroupedConvBlock, self).__init__()
        self.num_outputs = num_outputs
        self.ncpg = num_channels_per_group

        def get_norm_layer(num_outputs, total_ch):
            if norm_type == 'batch_norm':
                return nn.BatchNorm2d(total_ch)
            elif norm_type == 'group_norm':
                return nn.GroupNorm(num_outputs, total_ch)
            else:
                raise ValueError()

        total_ch = num_channels_per_group * num_outputs
        self.conv = nn.Sequential(
            conv1x1(num_inputs, total_ch),
            get_norm_layer(num_outputs, total_ch),
            nn.ReLU(),

            conv3x3(total_ch, total_ch, groups=num_outputs),
            get_norm_layer(num_outputs, total_ch),
            nn.ReLU(),

            conv3x3(total_ch, total_ch, groups=num_outputs),
            get_norm_layer(num_outputs, total_ch),
            nn.ReLU())

    def forward(self, x):
        x = self.conv(x)
        x_list = []
        for i in range(self.num_outputs):
            x_list.append(x[:, i * self.ncpg : (i+1) * self.ncpg])
        return torch.stack(x_list, dim=1)
