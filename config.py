import argparse

from utils import Config


def parse_train_args():
    parser = argparse.ArgumentParser()

    # general configurations
    parser.add_argument('--model_name', type=str, default='debug')
    parser.add_argument('--ckpt_root', type=str, default='ckpts', help='the root directory to save all the checkpoints')
    parser.add_argument('--selected_defects', type=str, default='0123456')
    parser.add_argument('--sat_defect_id', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--epoches', type=int, default=100)
    parser.add_argument('--num_final_ckpts', type=int, default=10)
    parser.add_argument('--save_ema_models', action='store_true')
    parser.add_argument('--ema_alpha', type=float, default=0.999)

    # image preprocessing
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--crop_size', type=int, default=0)
    parser.add_argument('--resize_mode', type=str, default='area', help='area or linear')
    parser.add_argument('--resize_backend', type=str, default='opencv', help='opencv, torch or pillow')
    parser.add_argument('--crop_method', type=str, default='center', help='center or random')

    # training data
    parser.add_argument('--train_img_dir', type=str, default='dataset/train')
    parser.add_argument('--train_csv_file', type=str, default='data/train/defect_training_gt_new.csv')
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--balance_data', type=int, default=1, help='0, 1 or 2 (0 to disable data balancing)')

    # testing data
    parser.add_argument('--test_img_dir', type=str, default='dataset/test')
    parser.add_argument('--test_csv_file', type=str, default='data/test/defect_testing_gt_new.csv')
    parser.add_argument('--test_batch_size', type=int, default=32)

    # general network configurations
    parser.add_argument('--net_type', type=str, default='shufflenet', help='shufflenet, shufflenet_v2 or resnet')
    parser.add_argument('--logits_act_type', type=str, default='sigmoid', help='sigmoid or softmax')
    parser.add_argument('--load_pretrained', action='store_true', help='whether to use pretrained model or randomly initialized model')
    parser.add_argument('--ckpt_path', type=str, default='', help='leave blank to use imagenet pretrained weight')
    parser.add_argument('--dropout_rate', type=float, default=0.5)
    parser.add_argument('--global_pooling_mode', type=str, default='average', help='average or combined')

    # shufflenet configurations
    parser.add_argument('--groups', type=int, default=8)
    parser.add_argument('--trunc_stage', action='store_true')

    # shufflenet_v2 configurations
    parser.add_argument('--width_mult', type=float, default=0.5, help='0.5 or 1.0')

    # resnet configurations
    parser.add_argument('--num_layers', type=int, default=18, help='18, 34, 50, 101 or 152')

    # prediction head - hybrid
    parser.add_argument('--add_hybrid_dataset', action='append', help='append additional dataset for hybrid training, must be in the format of [img_dir]:[csv_file]')
    parser.add_argument('--hybrid_test_id', type=int, default=-1, help='which of the hybrid heads should be used at testing time')

    # prediction head - softmax
    parser.add_argument('--num_softmax_bins', type=int, default=11, help='number of softmax bins for every defect except saturation')
    parser.add_argument('--sat_num_softmax_bins', type=int, default=21, help='number of softmax bins for saturation')

    # prediction head - convolution grouped by defects
    parser.add_argument('--num_channels_per_group', type=int, default=0, help='a positive integer for the separated grouped convolution as part of prediction head')

    # optimization
    parser.add_argument('--sat_loss_weight', type=float, default=1.0, help='currently only works for the regression model')
    parser.add_argument('--optimizer_type', type=str, default='sgdm', help='sgdm or adam')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--pretrain_lr', type=float, default=1e-4, help='learning rate for pretrained weights')
    parser.add_argument('--randinit_lr', type=float, default=1e-3, help='learning rate for randomly initialized weights')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--use_lr_decay', action='store_true', help='whether to use exponential decay on learning rate')
    parser.add_argument('--lr_decay_interval', type=int, default=10, help='number of epoches between every decay')
    parser.add_argument('--lr_decay_rate', type=float, default=0.95, help='exponential decay rate')
    parser.add_argument('--lr_decay_min', type=float, default=0, help='minimum learning rate of decay')

    # others
    parser.add_argument('--val_interval', type=int, default=50, help='number of iterations between every runtime validation')
    parser.add_argument('--ckpt_interval', type=int, default=10, help='number of epoches between every checkpoint saving')
    parser.add_argument('--model_restore_err_handling', type=str, default='notify', help='notify, raise or suppress')
    parser.add_argument('--tb_log_dir', type=str, default='runs', help='root directory to save the tensorboard records')
    parser.add_argument('--print_network', action='store_true', help='whether to print out network architecture, useful for debugging')
    parser.add_argument('--save_input_images', action='store_true', help='whether to show sampled input images on tensorboard')

    # DEBUG
    parser.add_argument('--try_deep_shallow_network', action='store_true')
    parser.add_argument('--normalize_head', action='store_true')
    parser.add_argument('--siamese_init', action='store_true')

    return Config(vars(parser.parse_args()))


def parse_eval_args():
    parser = argparse.ArgumentParser()

    # required arguments
    parser.add_argument('--model_name', type=str, default='debug')
    parser.add_argument('--ckpt_root', type=str, default='ckpts')
    parser.add_argument('--config_file', type=str, default='config.json')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--test_img_dir', type=str,
                        default='dataset/test', help='path to testing images')
    parser.add_argument('--test_csv_file', type=str,
                        default='data/test/defect_testing_gt_new.csv', help='path to testing labels')
    parser.add_argument('--test_batch_size', type=int, default=32)
    parser.add_argument('--obj_img_dir', type=str,
                        default='dataset/objective_testing_mod')
    parser.add_argument('--sbj_img_dir', type=str,
                        default='dataset/subjective_testing_public')
    parser.add_argument('--summary_version', type=int, default=1)
    parser.add_argument('--csv_save_dir', type=str, default='results')

    # optional arguments
    parser.add_argument('--test_spearman', action='store_true')
    parser.add_argument('--test_objective', action='store_true')
    parser.add_argument('--test_subjective', action='store_true')
    parser.add_argument('--use_ema_model', action='store_true')
    parser.add_argument('--use_averaged_weight', action='store_true')
    parser.add_argument('--record_suffix', type=str, default='')

    # override arguments (which will replace the settings in training config)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--hybrid_test_id', type=int, default=-1, help='which of the hybrid heads should be used at testing time')

    # DEBUG
    parser.add_argument('--print_network', action='store_true')
    parser.add_argument('--use_bilinear_fast_resize', action='store_true')

    return Config(vars(parser.parse_args()))


def parse_config(config):
    # convert selected_defects from string to a list of integers
    config.selected_defects = list(map(int, list(config.selected_defects)))

    # find out the index of saturation in the selected_defects
    if config.sat_defect_id in config.selected_defects:
        config.sat_idx = config.selected_defects.index(config.sat_defect_id)
    else:
        config.sat_idx = None

    # check if hybrid training is required
    if config.add_hybrid_dataset:
        config.hybrid_dataset = [{
            'img_dir': config.train_img_dir,
            'csv_file': config.train_csv_file
        }]
        for dataset_str in config.add_hybrid_dataset:
            dataset_params = dataset_str.split(':')
            config.hybrid_dataset.append({'img_dir': dataset_params[0],
                                          'csv_file': dataset_params[1]})
        config.num_hybrids = len(config.hybrid_dataset)

        assert config.hybrid_test_id < config.num_hybrids

    print(config)
    return config