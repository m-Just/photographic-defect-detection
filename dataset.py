import csv
from collections import OrderedDict
from collections.abc import Iterable

import cv2
import numpy as np
from PIL import Image
from torch.utils import data
import torchvision.transforms as transforms

from utils import select_by_indices, get_images_recursively


def read_annos(csv_file):
    annos = dict()
    with open(csv_file) as f:
        reader = csv.reader(f)
        next(reader) # skip header
        for r in reader:
            img_file = r[0].split('/')[-1]
            annos[img_file] = np.array(list(map(float, r[1:])), dtype=np.float32)
    return annos


def pil_image_in_ndarray(func):
    def convert(img):
        np_img = np.asarray(img, dtype=np.float32)
        np_img = func(np_img)
        np_img = np.round(np_img).astype(np.uint8)
        return Image.fromarray(np_img)
    return convert


def get_resize_func(min_size, max_size, mode, backend):
    if not isinstance(min_size, int) or not isinstance(max_size, int):
        raise ValueError()

    if min_size == max_size:
        def get_size(): return min_size
    elif min_size < max_size:
        def get_size(): return np.random.randint(min_size, max_size + 1)
    else:
        raise ValueError()

    if backend == 'opencv':
        mode_dict = {
            'area': cv2.INTER_AREA,
            'bilinear': cv2.INTER_LINEAR
        }
        if mode in mode_dict:
            mode = mode_dict[mode]
            def _resize(img):
                _size = get_size()
                return cv2.resize(img, (_size, _size), interpolation=mode)
            return _resize
        else:
            raise NotImplementedError()
    elif backend == 'torch':
        raise NotImplementedError()
    elif backend == 'pillow':
        raise NotImplementedError()
    else:
        raise ValueError()


def get_transform_params(config, mode):
    if mode == 'train':
        transform_params = {
            'input_size': config.input_size,
            'resize_mode': config.resize_mode,
            'resize_backend': config.resize_backend,
            'crop_size': config.crop_size,
            'crop_method': config.crop_method
        }
    elif mode == 'test':
        transform_params = {
            'input_size': config.input_size,
            'resize_mode': config.resize_mode,
            'resize_backend': config.resize_backend,
            'crop_size': config.crop_size,
            'crop_method': 'center'
        }
    else:
        raise ValueError()
    return transform_params


class DatasetFrame(data.Dataset):
    def __init__(self, img_dir, csv_file, selected_defects, transform_params,
                 label_parsing_func=None):
        self.selected_defects = selected_defects
        self.transform = self.get_transform(**transform_params)

        self.image_paths = get_images_recursively(img_dir)
        self.image_files = [p.split('/')[-1] for p in self.image_paths]

        if csv_file:
            self.annos = read_annos(csv_file)
            self.annos = select_by_indices(self.annos, selected_defects, dim=0)
            if label_parsing_func:
                self.annos = {k: label_parsing_func(v) for k, v in self.annos.items()}
        else:
            self.annos = None

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        pass

    def load_img(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert(mode='RGB')
        img = self.transform(img)
        return img, self.image_files[idx]

    @staticmethod
    def get_transform(input_size, resize_mode, resize_backend, crop_size, crop_method):
        transform = []

        if crop_size == input_size:
            min_resize = input_size
            max_resize = 256    # ad-hoc
        elif crop_size < input_size:
            min_resize = input_size
            max_resize = input_size
        else:
            raise ValueError()

        if resize_backend == 'opencv':
            resize_func = get_resize_func(min_resize, max_resize, resize_mode, resize_backend)
            resize_func = pil_image_in_ndarray(resize_func)
        elif resize_backend == 'torch':
            raise NotImplementedError()
        elif resize_backend == 'pillow':
            raise NotImplementedError()

        transform.append(resize_func)

        if not crop_size:
            crop_func = None
        elif crop_method == 'center':
            crop_func = transforms.CenterCrop(crop_size)
        elif crop_method == 'random':
            crop_func = transforms.RandomCrop(crop_size)
        else:
            raise ValueError()

        if crop_func is not None:
            transform.append(crop_func)

        transform.append(transforms.RandomHorizontalFlip())
        transform.append(transforms.ToTensor())

        return transforms.Compose(transform)


class TrainSet(DatasetFrame):
    def __getitem__(self, idx):
        img, img_file = self.load_img(idx)
        annotation = self.annos[img_file]
        loss_mask = np.ones(len(self.selected_defects), dtype=np.float32)

        data_dict = {
            'image': img,
            'label': annotation,
            'loss_mask': loss_mask,
            'img_path': self.image_paths[idx]
        }

        return data_dict

    def get_data_weights(self, version):
        if version == 1:
            print('Using data sampling v1')
            return self._get_data_weights_v1()
        elif version == 2:
            print('Using data sampling v2')
            return self._get_data_weights_v2()
        else:
            raise ValueError()

    def _get_data_weights_v1(self):
        weights = [1.0] * len(self.image_files)
        count = [0] * len(self.selected_defects)
        thres = [0.4] * len(self.selected_defects)
        weight_settings = [10.0, 10.0, 10.0, 20.0, 20.0, 20.0, 5.0]
        weight_settings = select_by_indices(weight_settings, self.selected_defects, dim=0)
        for j, img_file in enumerate(self.image_files):
            annot = self.annos[img_file]
            for i, score in enumerate(annot):
                # if an annotation has attributes larger than thres, we set the largest weight of that attribute to it
                if abs(float(score)) > thres[i]:
                    count[i] += 1
                    weights[j] = max(weight_settings[i], weights[j])
        return weights

    def _get_data_weights_v2(self):
        weights = [0.0] * len(self.image_files)
        weight_mult = [1.0] * len(self.selected_defects)
        for j, img_file in enumerate(self.image_files):
            annot = self.annos[img_file]
            for i, score in enumerate(annot):
                weights[j] += np.abs(score) * weight_mult[i]
        return weights


class TestSet(DatasetFrame):
    def __getitem__(self, idx):
        img, img_file = self.load_img(idx)
        loss_mask = np.ones(len(self.selected_defects), dtype=np.float32)

        data_dict = {
            'image': img,
            'loss_mask': loss_mask,
            'img_path': self.image_paths[idx]
        }

        if self.annos is not None:
            data_dict['label'] = self.annos[img_file]

        return data_dict


class HybridSet(data.Dataset):
    def __init__(self, *datasets):
        self.num_datasets = len(datasets)
        self.datasets = datasets
        self.image_files = []
        self.hybrid_ids = []
        self.image_indices = []
        for id, dataset in enumerate(self.datasets):
            assert self.datasets[0].selected_defects == dataset.selected_defects
            num_images = len(dataset.image_files)
            self.image_files += dataset.image_files
            self.hybrid_ids += [id] * num_images
            self.image_indices += list(range(num_images))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        hybrid_id = self.hybrid_ids[idx]
        dataset = self.datasets[hybrid_id]
        data_dict = dataset.__getitem__(self.image_indices[idx])
        data_dict['hybrid_id'] = hybrid_id
        return data_dict

    def get_data_weights(self, version):
        weights = []
        for id, dataset in enumerate(self.datasets):
            weights += dataset.get_data_weights(version)
        return weights


class DynamicHybridSet(data.Dataset):
    def __init__(self, BaseDataset, dataset_params):
        self.num_datasets = len(dataset_params)
        self.dataset_params = dataset_params
        self.datasets = OrderedDict()

        for name, params in self.dataset_params.items():
            self.datasets[name] = BaseDataset(**params)

        self.all_names = list(self.datasets.keys())
        self.hybrid_stream()

    def __len__(self):
        return sum([len(self.datasets[n]) for n in self.visible_names])

    def __getitem__(self, idx):
        for name in self.visible_names:
            if idx < self.end_idx[name]:
                break
        dataset = self.datasets[name]
        data_dict = dataset.__getitem__(idx - self.end_idx[name])
        data_dict['hybrid_dataset_name'] = name
        data_dict['hybrid_id'] = self.all_names.index(name)
        return data_dict

    def __iter__(self):
        for name in self.all_names:
            yield self.datasets[name]

    # def get_data_weights(self):
    #     hybrid_weights = []
    #     for name in self.visible_names:
    #         weights = self.datasets[name].get_data_weights()
    #         hybrid_weights.extend(weights)
    #     return hybrid_weights
    #
    # def get_data_weights_v2(self):
    #     hybrid_weights = []
    #     for name in self.visible_names:
    #         weights = self.datasets[name].get_data_weights_v2()
    #         hybrid_weights.extend(weights)
    #     return hybrid_weights
    #
    # def register_sampler(self, sampler):
    #     self.sampler = sampler

    # def set_weights(self, name, weights):
    #     self.weights[name] = weights
    #     hybrid_weights = []
    #     for name in self.visible_names:
    #         hybrid_weights.extend(self.weights[name])
    #     num_samples = len(hybrid_weights)
    #     self.sampler
    #
    # def set_sampler_params(self, balance_setting):
    #     hybrid_weights = []
    #     for name in self.visible_names:
    #         dataset = self.datasets[name]
    #         weights = get_balanced_weights(dataset, balance_setting)
    #         hybrid_weights.extend(weights)
    #     num_samples = len(hybrid_weights)
    #
    # def sampler(self, name, weights):
    #     self.weights[name] = weights
    #     hybrid_weights = []
    #     for name in self.visible_names:
    #         hybrid_weights.extend(self.weights[name])
    #     new_sampler = WeightedRandomSampler(weights, len(weights))
    #     self.sampler.__dict__ = new_sampler.__dict__.copy()
    #     #
    #     # self.samplers = OrderedDict()
    #     # for name, dataset in self.datasets.items():
    #     #     weights = dataset.
    #     #     self.samplers[name] =
    #     pass

    def hybrid_stream(self, names=None):
        if names is None:
            self.visible_names = self.all_names
        elif isinstance(names, Iterable):
            for n in names:
                if n not in self.all_names:
                    raise KeyError()
            self.visible_names = names
        else:
            raise ValueError()

        self.end_idx = dict()
        last_end_idx = 0
        for n in self.visible_names:
            self.end_idx[n] = len(self.datasets[n]) + last_end_idx
            last_end_idx = self.end_idx[n]

    def single_stream(self, name):
        self.hybrid_stream([name])
