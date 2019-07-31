import math
import copy
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.backbone.shufflenet import shufflenet
from networks.backbone.shufflenet_v2 import shufflenet_v2
from networks.backbone.resnet import resnet
from networks.pred_heads import SimpleLinearHead, SeparatedHead
from networks.pred_heads import HybridHead
from networks.building_blocks import GroupedConvBlock
from utils import ema_over_state_dict, config_override

__all__ = ['Network']


def get_linear_head(config, num_inputs, num_outputs, is_sat):
    if config.logits_act_type in ['none', 'sigmoid']:
        head = SimpleLinearHead(num_inputs, num_outputs,
                                dropout_rate=config.dropout_rate)
    elif config.logits_act_type == 'softmax':
        num_bins = config.sat_num_softmax_bins if is_sat else config.num_softmax_bins
        head = SimpleLinearHead(num_inputs, num_bins,
                                dropout_rate=config.dropout_rate)
    else:
        raise ValueError('Unrecognized activation type: '
                         + str(config.logits_act_type))
    return head


def get_heads_by_defect(config, num_inputs):
    head_list = []
    for defect_idx in config.selected_defects:
        is_sat = defect_idx == config.sat_idx
        head = get_linear_head(config, num_inputs, 1, is_sat)
        head_list.append(head)
    return head_list


class Network(nn.Module):
    @config_override
    def __init__(self, config):
        super(Network, self).__init__()
        self.global_pooling_mode = config.global_pooling_mode

        # define backbone network
        if config.net_type == 'shufflenet':
            self.backbone = shufflenet(
                groups=config.groups, trunc_stage=config.trunc_stage)
            self.pretrained_path = 'pretrained/shufflenet.pth.tar'
        elif config.net_type == 'shufflenet_v2':
            self.backbone = shufflenet_v2(config.width_mult)
            self.pretrained_path = f'pretrained/shufflenetv2_x{config.width_mult:.1f}.pth'
        elif config.net_type == 'resnet':
            self.backbone = resnet(config.num_layers)
            self.pretrained_path = f'pretrained/resnet{config.num_layers}.pth'
        else:
            raise ValueError()

        num_inputs = self.backbone.num_output_channels
        print(f'Number of backbone output channels: {num_inputs}')

        # define intermediate layers between backbone and head
        randinit_params = []

        self.neck = None
        if config.num_channels_per_group:
            self.neck = GroupedConvBlock(
                num_inputs, len(config.selected_defects),
                config.num_channels_per_group, config.gc_norm_type)
            randinit_params.append(self.neck.parameters())
            num_inputs = config.num_channels_per_group

        # define prediction head
        head = SeparatedHead(*get_heads_by_defect(config, num_inputs))
        if config.num_hybrids:
            head = HybridHead(head, config.num_hybrids, config.hybrid_test_id)
        self.head = head
        randinit_params.append(self.head.parameters())

        # restore model
        if config.load_pretrained:
            if config.ckpt_path:    # restore from history checkpoints
                self.restore(self, config.ckpt_path,
                             err_handling=config.model_restore_err_handling)
                self.pretrain_params = self.parameters()
                self.randinit_params = None
                print(f'Successfully loading checkpoint {config.ckpt_path}')
            else:                   # restore imagenet-pretrained backbone
                self.restore(self.backbone, self.pretrained_path,
                             err_handling=config.model_restore_err_handling)
                self.pretrain_params = self.backbone.parameters()
                self.randinit_params = chain(*randinit_params)
                print('Successfully loading imagenet-pretrained model')
        else:
            self.pretrain_params = None
            self.randinit_params = self.parameters()

        # initialize ema model state dict
        if config.save_ema_models:
            self.ema_state_dict = copy.deepcopy(self.state_dict())
            self.ema_alpha = config.ema_alpha
        else:
            self.ema_state_dict = None

    @staticmethod
    def get_ckpt_path(model_root, epoch, ema=False):
        ckpt_path = f'{model_root}/epoch-{epoch}'
        if ema:
            ckpt_path += '-ema'
        ckpt_path += '.pkl'
        return ckpt_path

    @staticmethod
    def save(state_dict, ckpt_path):
        save_dict = dict()
        for k, v in state_dict.items():
            save_dict[k[7:] if k.startswith('module') else k] = v
        torch.save(save_dict, ckpt_path)

    @staticmethod
    def restore(module, ckpt_path, err_handling='notify'):
        state_dict = module.state_dict()
        pretrained_dict = torch.load(ckpt_path)
        pretrained_dict = {k[7:] if k.startswith('module') else k: v
                           for k, v in pretrained_dict.items()}

        try:
            module.load_state_dict(pretrained_dict)
        except Exception as e:
            if err_handling == 'notify':
                print(e)
            elif err_handling == 'raise':
                raise e
            elif err_handling == 'suppress':
                pass
            else:
                raise ValueError()

        pretrained_dict = {k: v for k,
                           v in pretrained_dict.items() if k in state_dict}
        module.load_state_dict(pretrained_dict)

    def save_to(self, model_root, epoch):
        ckpt_path = self.get_ckpt_path(model_root, epoch, ema=False)
        self.save(self.state_dict(), ckpt_path)
        if self.ema_state_dict is not None:
            ema_ckpt_path = self.get_ckpt_path(model_root, epoch, ema=True)
            self.save(self.ema_state_dict, ema_ckpt_path)

    def update_ema_weights(self):
        ema_over_state_dict(self.ema_state_dict,
                            self.state_dict(), self.ema_alpha)

    def forward(self, x, hybrid_id=None):
        x = self.backbone(x)

        if self.neck is not None:
            if isinstance(self.neck, GroupedConvBlock):
                x = self.neck(x)
            else:
                raise RuntimeError()

        # GroupedConvBlock produces 5D outputs, however adaptive_avg_pool2d
        # only accept 4D inputs, thus we need to reshape the tensor into 4D
        batch_size, sizes = x.size(0), x.size()[1:-2]
        x = x.view(batch_size, -1, x.size(-2), x.size(-1))

        # convert feature maps into vectors by pooling
        if self.global_pooling_mode == 'average':
            x = F.adaptive_avg_pool2d(x, (1, 1))
        elif self.global_pooling_mode == 'combined':
            x_avg = F.adaptive_avg_pool2d(x, (1, 1))
            x_max = F.adaptive_max_pool2d(x, (1, 1))
            x = (x_avg + x_max) / 2
        else:
            spatial_size = x.size(-1)
            s = round(math.sqrt(spatial_size))
            if self.global_pooling_mode == 'avgofmax':
                x = F.adaptive_max_pool2d(x, (s, s))
                x = F.adaptive_avg_pool2d(x, (1, 1))
            elif self.global_pooling_mode == 'maxofavg':
                x = F.adaptive_avg_pool2d(x, (s, s))
                x = F.adaptive_max_pool2d(x, (1, 1))
            else:
                raise ValueError()

        # restore the dimensions previous merged
        assert x.size(-1) == 1 and x.size(-2) == 1
        x = x.view(batch_size, *sizes)

        # feature vectors into final outputs
        if isinstance(self.head, HybridHead):
            x = self.head(x, h=hybrid_id)
        else:
            x = self.head(x)

        return x


# DEBUG
class DeepShallowNetwork(Network):
    def __init__(self, config, deep_net, shallow_net):
        super(Network, self).__init__()
        self.normalize_head = config.normalize_head
        self.deep_net = deep_net
        self.shallow_net = shallow_net
        self.pretrain_params = list(
            deep_net.pretrain_params) + list(shallow_net.pretrain_params)
        self.randinit_params = list(
            deep_net.randinit_params) + list(shallow_net.randinit_params)

        num_inputs = deep_net.backbone.num_output_channels + shallow_net.backbone.num_output_channels
        head_list = get_heads_by_defect(config, num_inputs)
        self.head = SeparatedHead(*head_list)

    def forward(self, x, _=None):
        x_d = self.deep_net.backbone(x)
        x_s = self.shallow_net.backbone(x)
        if self.normalize_head:
            x_d = x_d / x_d.detach().abs().mean(dim=-1).unsqueeze(-1)
            x_s = x_s / x_s.detach().abs().mean(dim=-1).unsqueeze(-1)
        x = torch.cat([x_d, x_s], dim=-1)
        return self.head(x)


if __name__ == '__main__':  # used for debug
    from config import Config
    config = Config()
    config.net_type = 'shufflenet'
    config.groups = 8
    config.global_pooling_mode = 'average'
    config.trunc_stage = True
    config.selected_defects = '0123456'
    config.num_channels_per_group = 128
    config.num_hybrids = 3
    config.logits_act_type = 'sigmoid'
    config.dropout_rate = 0.5
    config.load_pretrained = True
    config.ckpt_path = ''

    config.net_type = 'resnet'
    config.num_layers = 101

    config.net_type = 'shufflenet_v2'
    config.width_mult = 0.5
    network = Network(config)
    print(network)
