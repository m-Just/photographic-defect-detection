import copy

import torch
import torch.nn as nn

from networks.building_blocks import conv3x3, conv1x1

__all__ = ['SimpleLinearHead', 'HybridHead']


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
        '''
        Args:
            x: a tensor which dimension can be 2 or 3. If dimension is 2 then
                all heads share the same input x, otherwise, the i-th head
                uses x[:, i] as input.
        Returns:
            outputs of each head concatenated at dim=1.
        '''
        x_list = []
        for i in range(self.num_outputs):
            head = getattr(self, f'output_{i}')
            x_ = head(x[:, i] if x.dim() == 3 else x)
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

        if not isinstance(basic_head, nn.Module):
            raise TypeError()
        if num_hybrids < 2:
            raise ValueError()

        for h in range(num_hybrids):
            setattr(self, f'hybrid_{h}', basic_head)
            basic_head = copy.deepcopy(basic_head)
        del basic_head  # remove the last unused duplicate

    def forward(self, x, h=None):
        if h is None and isinstance(self.constant_id, int):
            h = [self.constant_id] * x.size(0)

        x_list = []
        for i in range(x.size(0)):
            layer = getattr(self, f'hybrid_{h[i]}')
            x_ = layer(x[i].unsqueeze(0))
            x_list.append(x_)
        return torch.cat(x_list, dim=0)
