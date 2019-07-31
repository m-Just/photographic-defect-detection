import argparse

import torch
import torch.nn as nn

from model import Network
from model_wrap import get_model_wrap
from networks.building_blocks import DummyModule
from dataloader import get_test_dataloader
from utils import Config, CSV_Writer
from utils import avg_over_state_dict, determine_device, get_defect_names_by_idx
from utils import makedirs_if_not_exists
from config import parse_eval_args, parse_config
from metric import compute_ranking_accuracy


def fuse(conv, bn):
    w = conv.weight
    mean = bn.running_mean
    var_sqrt = torch.sqrt(bn.running_var + bn.eps)

    beta = bn.weight
    gamma = bn.bias

    if conv.bias is not None:
        b = conv.bias
    else:
        b = mean.new_zeros(mean.shape)

    w = w * (beta / var_sqrt).reshape([conv.out_channels, 1, 1, 1])
    b = (b - mean)/var_sqrt * beta + gamma

    fused_conv = nn.Conv2d(conv.in_channels,
                           conv.out_channels,
                           conv.kernel_size,
                           conv.stride,
                           conv.padding,
                           groups=conv.groups,
                           bias=True)
    fused_conv.weight = nn.Parameter(w)
    fused_conv.bias = nn.Parameter(b)
    return fused_conv


def fuse_conv_bn(m):
    children = list(m.named_children())
    c = None
    cn = None
    for name, child in children:
        if isinstance(child, nn.BatchNorm2d):
            bc = fuse(c, child)
            m._modules[cn] = bc
            m._modules[name] = DummyModule()
            c = None
        elif isinstance(child, nn.Conv2d):
            c = child
            cn = name
        else:
            fuse_conv_bn(child)


def write_spearman_to_csv(corr, selected_defects, model_name, save_dir):
    csv_path = f'{save_dir}/evaluation_results_spearman.csv'
    header = ['Model Name', 'Overall'] + \
        get_defect_names_by_idx(selected_defects)
    with CSV_Writer(csv_path, header, mode='a') as writer:
        writer.writerow([model_name] + corr)


def write_ranking_summary_to_csv(rates, model_name, mode, save_dir):
    csv_path = f'{save_dir}/evaluation_results_{mode}.csv'
    header = ['Model Name'] + list(rates.keys())
    summary_scores = [r[-1] for r in list(rates.values())]
    with CSV_Writer(csv_path, header, mode='a') as writer:
        writer.writerow([model_name] + summary_scores)


def main(config):
    device = determine_device()

    if config.use_averaged_weight:
        ckpt_paths = []
        for i in range(config.num_final_ckpts):
            ckpt_path = Network.get_ckpt_path(
                model_root, config.epoch - i, ema=config.use_ema_model)
            ckpt_paths.append(ckpt_path)
        _, config.ckpt_path = avg_over_state_dict(ckpt_paths)
    else:
        config.ckpt_path = Network.get_ckpt_path(
            model_root, config.epoch, ema=config.use_ema_model)

    model = Network(config)
    if config.fuse_conv_bn:
        fuse_conv_bn(model)
        Network.save(model.state_dict(), f'{config.ckpt_path[:-4]}_fused.pkl')
    model.eval()
    model = model.to(device)
    if config.print_network:
        print(model)

    wrap = get_model_wrap(config, model, None, device)

    record_name = config.model_name
    record_name += f'_epoch{config.epoch}'
    if config.use_ema_model:
        record_name += '_ema'
    if config.fuse_conv_bn:
        record_name += '_fused'
    if config.record_suffix:
        record_name += f'_{config.record_suffix}'

    makedirs_if_not_exists(config.csv_save_dir)

    # evaluate spearman correlation
    if config.test_spearman:
        test_dataloader = get_test_dataloader(config)
        _, corr = wrap.validate(test_dataloader)
        print(corr)
        write_spearman_to_csv(corr, config.selected_defects, record_name, config.csv_save_dir)

    # evaluate on objective testing set
    if config.test_objective:
        mode = 'obj'
        test_dataloader = get_test_dataloader(
            config, test_img_dir=config.obj_img_dir, test_csv_file=None,
            test_batch_size=1)
        score_dict, _, _ = wrap.gather(test_dataloader, gather_loss=False)
        rates = compute_ranking_accuracy(config, score_dict, mode=mode)
        write_ranking_summary_to_csv(rates, record_name, mode, config.csv_save_dir)

    # evaluate on subjective testing set
    if config.test_subjective:
        mode = 'sbj'
        test_dataloader = get_test_dataloader(
            config, test_img_dir=config.sbj_img_dir, test_csv_file=None,
            test_batch_size=1)
        score_dict, _, _ = wrap.gather(test_dataloader, gather_loss=False)
        rates = compute_ranking_accuracy(
            config, score_dict, mode=mode, summary_version=config.summary_version)
        write_ranking_summary_to_csv(rates, record_name, mode, config.csv_save_dir)


if __name__ == '__main__':
    # parse command-line arguments
    eval_config = parse_eval_args()

    # load configurations from training time
    model_root = f'{eval_config.ckpt_root}/{eval_config.model_name}'
    config_path = f'{model_root}/{eval_config.config_file}'
    config = Config({k: v for k, v in Config(config_path).items()})

    # update the training configurations with the new configurations
    config.update(vars(eval_config))

    # override specific configurations to avoid unwanted behaviors
    config.save_ema_models = False  # this is only used at training time

    # main program
    main(parse_config(config))
