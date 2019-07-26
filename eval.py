import argparse

from model import Network
from model_wrap import get_model_wrap
from dataloader import get_test_dataloader
from utils import Config, CSV_Writer
from utils import avg_over_state_dict, determine_device, get_defect_names_by_idx
from utils import makedirs_if_not_exists
from config import parse_eval_args, parse_config
from metric import compute_ranking_accuracy


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
                model_root, config.epoch - i, config.use_ema_model)
            ckpt_paths.append(ckpt_path)
        _, config.ckpt_path = avg_over_state_dict(ckpt_paths)
    else:
        config.ckpt_path = Network.get_ckpt_path(
            model_root, config.epoch, config.use_ema_model)

    model = Network(config)
    model.eval()
    model = model.to(device)

    wrap = get_model_wrap(config, model, None, device)

    record_name = config.model_name
    record_name += f'_epoch{config.epoch}'
    if config.use_ema_model:
        record_name += '_ema'
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
