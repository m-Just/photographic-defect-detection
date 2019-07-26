import copy

import torch
import torch.nn as nn

from networks.building_blocks import conv3x3, conv1x1

__all__ = ['SimpleLinearHead', 'HybridHead', 'GroupedConvHead']


def simple_linear_layer(num_inputs, num_outputs, dropout_rate=0.5):
    return nn.Sequential(nn.Dropout(dropout_rate),
                         nn.Linear(num_inputs, num_outputs))


class SimpleLinearHead(nn.Module):
    def __init__(self, num_inputs, num_outputs, dropout_rate=0.5):
        super(SimpleLinearHead, self).__init__()
        self.linear = simple_linear_layer(num_inputs, num_outputs, dropout_rate)

    def forward(self, x):
        x = self.linear(x)
        return x


class SeparatedHead(nn.Module):
    def __init__(self, *basic_heads):
        super(SeparatedHead, self).__init__()
        self.num_outputs = len(basic_heads)
        for i, head in enumerate(basic_heads):
            setattr(self, f'output_{i}', head)

    def forward(self, x):
        x_list = []
        for i in range(self.num_outputs):
            head = getattr(self, f'output_{i}')
            x_ = head(x)
            x_list.append(x_)
            assert x_.dim() == 2
        return torch.cat(x_list, dim=1)


class HybridHead(nn.Module):
    def __init__(self, basic_head, num_hybrids, constant_id=None):
        ''' Create a hybrid head by duplicating a provided basic head.
            Every duplicate will be associated with a unique hybrid id,
            represented by h, which should be passed in during forward time
            along with the module input x. If constant_id is set, h can be
            omitted at runtime.
        Args:
            basic_head: any torch.nn.Module
            num_hybrids: this determines how many duplicates will be created
            constant_id: override the runtime hybrid id by a constant value when
                no hybrid id is provided for self.forward, normally triggered
                when testing on a single dataset.
        '''
        super(HybridHead, self).__init__()
        self.constant_id = constant_id

        if not isinstance(basic_head, nn.Module) or num_hybrids < 2:
            raise ValueError()

        for h in range(num_hybrids):
            setattr(self, f'hybrid_{h}', basic_head)
            basic_head = copy.deepcopy(basic_head)
        del basic_head  # remove the last unused duplicate

    def forward(self, x, h=None):
        if self.constant_id is not None:
            h = [self.constant_id] * x.size(0)

        x_list = []
        for i in range(x.size(0)):
            layer = getattr(self, f'hybrid_{h[i]}')
            x_ = layer(x[i].unsqueeze(0))
            x_list.append(x_)
        return torch.cat(x_list, dim=0)


class GroupedConvHead(nn.Module):
    def __init__(self, basic_head, num_inputs, num_outputs, num_channels_per_group):
        super(GroupedConvHead, self).__init__()
        self.num_outputs = num_outputs
        self.ncpg = num_channels_per_group

        total_ch = num_channels_per_group * num_outputs
        self.conv = nn.Sequential(
            conv1x1(num_inputs, total_ch),
            nn.BatchNorm2d(total_ch),
            nn.ReLU(),

            conv3x3(total_ch, total_ch, groups=num_outputs),
            nn.BatchNorm2d(total_ch),
            nn.ReLU(),

            conv3x3(total_ch, total_ch, groups=num_outputs),
            nn.BatchNorm2d(total_ch),
            nn.ReLU())

        for n in range(num_outputs):
            setattr(self, f'fc_{n}', basic_head)
            basic_head = copy.deepcopy(basic_head)
        del basic_head  # remove the last unused duplicate

    def forward(self, x, hybrid_id=None):
        raise NotImplementedError()

        x = self.conv(x)

        x_group = []
        for n in range(self.num_outputs):
            x_input = x[:, n * self.ncpg : (n+1) * self.ncpg]
            fc = getattr(self, f'fc_{n}')
            # DEBUG
            # _x = HybridHead.forward_via(fc, x_input, hybrid_id)
            _x = fc(x_input, hybrid_id)

            x_group.append(_x)
        x = torch.squeeze(torch.stack(x_group, dim=1), dim=2)

        return x
