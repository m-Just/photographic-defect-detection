import torch
import torch.utils.data.sampler as sampler

from utils import config_override
from dataset import TrainSet, TestSet, HybridSet
from dataset import get_transform_params


def get_dataloader(dataset, balance_data, batch_size, num_workers):
    if balance_data:
        weights = dataset.get_data_weights(balance_data)
        sampler_ = sampler.WeightedRandomSampler(weights, len(weights))
    else:
        sampler_ = sampler.RandomSampler(dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=sampler_,
        num_workers=num_workers)

    return dataloader


@config_override
def get_train_dataloader(config):
    transform_params = get_transform_params(config, 'train')
    if config.num_hybrids:
        datasets = []
        for id in range(config.num_hybrids):
            dataset_params = config.hybrid_dataset[id]
            dataset = TrainSet(dataset_params['img_dir'], dataset_params['csv_file'],
                               config.selected_defects, transform_params)
            datasets.append(dataset)
        dataset = HybridSet(*datasets)
    else:
        dataset = TrainSet(config.train_img_dir, config.train_csv_file,
                           config.selected_defects, transform_params)
    dataloader = get_dataloader(dataset, config.balance_data,
                                config.train_batch_size, config.num_workers)
    return dataloader


@config_override
def get_test_dataloader(config):
    transform_params = get_transform_params(config, 'test')
    dataset = TestSet(config.test_img_dir, config.test_csv_file,
                      config.selected_defects, transform_params)
    dataloader = get_dataloader(dataset, 0, config.test_batch_size,
                                config.num_workers)
    return dataloader


# DEBUG
def get_ranking_dataloader(config):
    real_world_set = TestSet(img_dir='ext/xiaomi/st_test',
                             csv_file=None,
                             selected_defects=config.selected_defects,
                             transform_params=get_transform_params(config, 'test'))
    real_world_loader = torch.utils.data.DataLoader(
        real_world_set, batch_size=1, num_workers=config.num_workers)
    return real_world_loader


def measure_dist_by_sampling(dataloader, num_epoches=1):
    labels = []
    for epoch in range(num_epoches):
        for i, data in enumerate(dataloader):
            label_batch = data['label']
            labels.append(label_batch)
    return torch.cat(labels, dim=0)