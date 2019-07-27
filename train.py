import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'    # for clearer CUDA debug message

import numpy as np
import torch.optim as optim

from config import parse_train_args, parse_config
from model_wrap import get_model_wrap
from dataloader import get_train_dataloader, get_test_dataloader, get_ranking_dataloader
from model import Network, DeepShallowNetwork
from utils import Config, TensorboardWriter, Timer
from utils import determine_device, at_interval
from utils import makedirs_if_not_exists, get_defect_names_by_idx

from metric import evaluate_blur_ranking_accuracy


def get_optimizer(method, default_lr, momentum,
                  pretrain_lr, pretrain_params,
                  randinit_lr, randinit_params):
    params_lr = []
    if pretrain_params:
        print('Initial lr for pretrained parameters '
              + f'(group #{len(params_lr)}): {pretrain_lr}')
        params_lr.append({'params': pretrain_params, 'lr': pretrain_lr})
    if randinit_params:
        print('Initial lr for randomly initialized parameters '
              + f'(group #{len(params_lr)}): {randinit_lr}')
        params_lr.append({'params': randinit_params, 'lr': randinit_lr})

    if method == 'sgdm':
        print('Using SGD optimizer with momentum')
        optimizer = optim.SGD(params_lr, lr=default_lr, momentum=momentum)
    elif method == 'adam':
        print('Using Adam optimizer')
        optimizer = optim.Adam(params_lr, lr=default_lr)
    else:
        raise NotImplementedError()
    return optimizer


def apply_lr_schedule(optimizer, lr_decay_rate, lr_decay_min):
    exponential_decay = lambda step: max(lr_decay_rate ** step, lr_decay_min)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, exponential_decay)
    return scheduler


def main(config):
    device = determine_device()

    # DEBUG
    if config.try_deep_shallow_network:
        import copy
        deep_config = copy.deepcopy(config)
        deep_config.trunc_stage = False
        deep_net = Network(deep_config)

        if config.siamese_init:
            shallow_config = copy.deepcopy(deep_config)
        else:
            shallow_config = copy.deepcopy(config)
            shallow_config.trunc_stage = True
        shallow_net = Network(shallow_config)

        model = DeepShallowNetwork(config, deep_net, shallow_net)
    else:
        model = Network(config)

    model.train()
    model = model.to(device)
    if config.print_network:
        print(model)

    optimizer = get_optimizer(config.optimizer_type, config.lr, config.momentum,
                              config.pretrain_lr, model.pretrain_params,
                              config.randinit_lr, model.randinit_params)
    if config.use_lr_decay:
        scheduler = apply_lr_schedule(
            optimizer, config.lr_decay_rate, config.lr_decay_min)

    wrap = get_model_wrap(config, model, optimizer, device)

    train_dataloader = get_train_dataloader(config)
    test_dataloader = get_test_dataloader(config)

    print(f'Train dataloader: {len(train_dataloader)} batches/epoch, batch_size={config.train_batch_size}')
    print(f'Test dataloader: {len(test_dataloader)} batches/epoch, batch_size={config.test_batch_size}')

    log_dir = f'./{config.tb_log_dir}/{config.model_name}'
    writer = TensorboardWriter(log_dir, get_defect_names_by_idx(config.selected_defects))

    # DEBUG
    # real_world_loader = get_ranking_dataloader(config)

    num_steps = 0
    for epoch in range(config.epoches):
        # DEBUG
        # with Timer('blur ranking'):
        #     acc_dict = evaluate_blur_ranking_accuracy(wrap, real_world_loader,
        #                                               config.selected_defects.index(5))
        # writer.add_dict('eval', acc_dict, num_steps)

        for i, data in enumerate(train_dataloader):
            # training iteration
            wrap.train(data)
            if config.save_ema_models:
                model.update_ema_weights()

            # DEBUG
            # if at_interval(i, 50, start_index=0):
            #     print('========= Model prediction sample ========')
            #     print(np.around(wrap.scores[0].data.cpu().numpy(), decimals=4))

            # runtime logging and evaluation
            if at_interval(i, config.val_interval, start_index=0):
                print(f'Epoch {epoch+1}/{config.epoches}, iteration {i}/{len(train_dataloader)}:')
                print(wrap.loss_dict)
                writer.add_loss_dict('train', wrap.loss_dict, num_steps)

                with Timer('validation'):
                    loss_val_dict, _ = wrap.validate(test_dataloader, compute_spearman=False)
                writer.add_loss_dict('val', wrap.loss_dict, num_steps)

                if config.save_input_images:
                    writer.add_image_tensor('input', wrap.inputs[0][0], num_steps)

            num_steps += config.train_batch_size

        # save checkpoint
        if at_interval(epoch, config.ckpt_interval) or epoch >= config.epoches - config.num_final_ckpts:
            model.save_to(model_root, epoch + 1)
            with Timer('computing spearman corr'):
                _, corr = wrap.validate(test_dataloader)
            print(corr)
            writer.add_spearman_rank_corr('spearman', corr, num_steps)

        # apply learning rate decay
        if config.use_lr_decay and at_interval(epoch, config.lr_decay_interval):
            scheduler.step()
            for n, param in enumerate(optimizer.state_dict()['param_groups']):
                print(f'Current lr for parameter group #{n} = {param["lr"]}')


if __name__ == '__main__':
    # parse command-line arguments
    config = parse_train_args()

    # save configurations
    model_root = f'{config.ckpt_root}/{config.model_name}'
    makedirs_if_not_exists(model_root)
    config.save(f'{model_root}/config.json')

    # main program
    main(parse_config(config))
