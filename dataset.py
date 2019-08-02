import csv
from collections import OrderedDict
from collections.abc import Iterable

import cv2
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from scipy.ndimage import gaussian_filter
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from torch.utils import data

from utils import select_by_indices, get_images_recursively


def read_annos(csv_file):
    annos = dict()
    with open(csv_file) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for r in reader:
            img_file = r[0].split('/')[-1]
            annos[img_file] = np.array(
                list(map(float, r[1:])), dtype=np.float32)
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
        raise TypeError()

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
    if mode not in ['train', 'test']:
        raise ValueError()
    training = mode == 'train'

    transform_params = {
        'input_size': config.input_size,
        'max_input_size': config.max_input_size if training else
                          config.input_size,
        'resize_mode': config.resize_mode,
        'resize_backend': config.resize_backend,
        'crop_size': config.crop_size,
        'crop_method': config.crop_method if training else 'center',
        'h_flip': training
    }

    return transform_params


class DatasetFrame(data.Dataset):
    def __init__(self, img_dir, csv_file, selected_defects, transform_params,
                 std_csv_file=None):
        self.img_dir = img_dir
        self.selected_defects = selected_defects
        self.transform_params = transform_params
        self.transform = self.get_transform(**transform_params)

        self.image_paths = get_images_recursively(img_dir)
        self.image_files = [p.split('/')[-1] for p in self.image_paths]

        if csv_file:
            self.annos = read_annos(csv_file)
            self.annos = select_by_indices(self.annos, selected_defects, dim=0)
        else:
            self.annos = None

        if std_csv_file:
            self.stds = read_annos(std_csv_file)
            self.stds = select_by_indices(self.stds, selected_defects, dim=0)
        else:
            self.stds = None

        self._default_loss_mask = np.ones(len(selected_defects), dtype=np.float32)

    def __len__(self):
        return len(self.image_files)

    def load_img(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert(mode='RGB')
        return img, self.image_files[idx]

    @staticmethod
    def get_transform(input_size, max_input_size, resize_mode, resize_backend,
                      crop_size, crop_method, h_flip):
        transform = []

        if input_size > crop_size >= 0:
            min_resize = input_size
            max_resize = max_input_size
        else:
            raise ValueError()

        if resize_backend == 'opencv':
            resize_func = get_resize_func(
                min_resize, max_resize, resize_mode, resize_backend)
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

        if h_flip:
            transform.append(transforms.RandomHorizontalFlip())
        transform.append(transforms.ToTensor())

        return transforms.Compose(transform)


class TrainSet(DatasetFrame):
    def __init__(self, *args, **kwargs):
        frame_kwargs = {}
        frame_kwargs['std_csv_file'] = kwargs['std_csv_file']
        super(TrainSet, self).__init__(*args, **frame_kwargs)

        self.use_augmentation = kwargs['use_augmentation']
        self.use_weighted_loss = kwargs['use_weighted_loss']
        if self.use_weighted_loss:
            self.compute_label_hist()

    def __getitem__(self, idx):
        img, img_file = self.load_img(idx)
        if self.use_augmentation:
            img, loss_mask = self.apply_random_augmentation(
                img, self.transform_params['input_size'])
            loss_mask = select_by_indices(
                loss_mask, self.selected_defects, dim=0)
            loss_mask = np.array(loss_mask, dtype=np.float32)
        else:
            loss_mask = self._default_loss_mask.copy()
        img = self.transform(img)

        annotation = self.annos[img_file]
        if self.use_weighted_loss:
            self.apply_weighted_loss(annotation, loss_mask)

        data_dict = {
            'image': img,
            'label': annotation,
            'loss_mask': loss_mask,
            'img_path': self.image_paths[idx]
        }

        if self.stds is not None:
            data_dict['std'] = self.stds[img_file]
        else:
            data_dict['std'] = np.zeros(
                len(self.selected_defects), dtype=np.float32)

        return data_dict

    # ------------------------- Weighted loss methods -------------------------
    def compute_label_hist(self):
        sat_bins = np.arange(-1.05, 1.05 + 1e-3, 0.1)
        other_bins = np.arange(-0.05, 1.05 + 1e-3, 0.1)

        self.hist_and_bins = []
        for i, d in enumerate(self.selected_defects):
            anno_by_defect = []
            for anno in self.annos.values():
                anno_by_defect.append(anno[i])
            bins = sat_bins if d == 2 else other_bins
            hist, bins = np.histogram(anno_by_defect, bins=bins)
            hist = hist.astype(np.float32)
            if self.use_weighted_loss == 1:
                hist = np.sqrt(hist)
            elif self.use_weighted_loss == 2:
                hist = np.sqrt(np.sqrt(hist))
            else:
                raise ValueError()
            self.hist_and_bins.append((hist, bins))

    def apply_weighted_loss(self, annotation, loss_mask):
        weight = []
        for i, anno in enumerate(annotation):
            for j, bin in enumerate(self.hist_and_bins[i][1]):
                if anno < bin: break
            hist_min = min(self.hist_and_bins[i][0])
            weight.append(hist_min / self.hist_and_bins[i][0][j-1])
        loss_mask *= np.array(weight, dtype=np.float32)

    # ------------------------- Data sampling methods -------------------------
    def get_data_weights(self, version):
        if version == 1:
            print(f'Using data sampling v1 for {self.img_dir}')
            return self._get_data_weights_v1()
        elif version == 2:
            print(f'Using data sampling v2 for {self.img_dir}')
            return self._get_data_weights_v2()
        else:
            raise ValueError()

    def _get_data_weights_v1(self):
        weights = [1.0] * len(self.image_files)
        count = [0] * len(self.selected_defects)
        thres = [0.4] * len(self.selected_defects)
        weight_settings = [10.0, 10.0, 10.0, 20.0, 20.0, 20.0, 5.0]
        weight_settings = select_by_indices(
            weight_settings, self.selected_defects, dim=0)
        for j, img_file in enumerate(self.image_files):
            annot = self.annos[img_file]
            for i, score in enumerate(annot):
                # if an annotation has attributes larger than thres,
                # we set the largest weight of that attribute to it
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

    # ------------------------- Augmentation methods --------------------------
    @staticmethod
    def apply_random_augmentation(image, input_size, augs=[0, 1, 2, 3, 4, 5, 6]):
        def get_random_range(minval, maxval):
            return np.random.rand(1) * (maxval - minval) + minval

        choice = np.random.choice(augs, 1)[0]

        if choice == 0:     # saturation
            factor = np.random.choice([0, 0.6, 0.7, 1.3, 1.4], 1,
                                      p=[0.1, 0.15, 0.3, 0.3, 0.15])[0]
            image = TF.adjust_saturation(image, factor)
            loss_mask = [1, 0, 0, 0, 1, 1, 1]

        elif choice == 1:   # rotation
            angle = np.random.rand(1) * 30 - 15
            image = TF.rotate(image, angle, resample=Image.BILINEAR)
            loss_mask = [0, 1, 1, 1, 1, 0, 0]

        elif choice == 2:   # resized crop
            scale = [0.6, 1.0]
            aspect_ratio = [3./4, 4./3]
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                image, scale, aspect_ratio)
            image = TF.resized_crop(image, i, j, h, w,
                                    size=(input_size, input_size), interpolation=2)  # bilinear
            loss_mask = [1, 1, 1, 1, 1, 1, 0]

        elif choice == 3:   # brightness
            if np.random.rand(1) > 0.5:
                factor = get_random_range(0.8, 1.2)
                image = TF.adjust_brightness(image, factor)
                loss_mask = [0, 1, 1, 0, 1, 1, 1]
            else:
                gamma = np.random.choice([1/1.2, 1.2], 1)[0]
                image = TF.adjust_gamma(image, gamma)
                loss_mask = [0, 1, 0, 0, 0, 1, 1]

        elif choice == 4:   # noise
            sigma = get_random_range(0.8, 2.)
            image = TrainSet.apply_gaussian_noise(image, sigma)
            loss_mask = [1, 1, 1, 0, 1, 1, 1]

        elif choice == 5:   # blur
            sigma = get_random_range(1.4, 2.0)
            image = TrainSet.apply_gaussian_blur(image, sigma)
            loss_mask = [1, 1, 1, 0, 1, 0, 1]

        elif choice == 6:   # white balance
            hue_factor = get_random_range(-0.25, 0.25)
            image = TF.adjust_hue(image, hue_factor)
            loss_mask = [1, 0, 1, 1, 1, 1, 1]

        return image, loss_mask

    @staticmethod
    def apply_gaussian_noise(image, sigma, mult=10.):
        '''
        Args:
            image: a PIL format image.
            sigma: the standard deviation of the normal distribution used for
                   noise sampling, of which the value should be within [0, +inf).
            mult: a multiplier to the magnitude of noise.
        Returns:
            The augmented image in PIL format.
        '''
        np_image = np.asarray(image, dtype=np.float32)
        np_image += np.random.normal(scale=sigma, size=np_image.shape) * mult
        np_image = np.clip(np_image, 0., 255.)
        return Image.fromarray(np.round(np_image).astype(np.uint8))

    @staticmethod
    def apply_gaussian_blur(image, sigma):
        '''
        Args:
            image: a PIL format image.
            sigma: the standard deviation for Gaussian kernel.
        Returns:
            The augmented image in PIL format.
        '''
        np_image = np.asarray(image, dtype=np.float32)
        np_image = gaussian_filter(np_image, sigma=[sigma, sigma, 0])
        return Image.fromarray(np.round(np_image).astype(np.uint8))


class TestSet(DatasetFrame):
    def __getitem__(self, idx):
        img, img_file = self.load_img(idx)
        loss_mask = self._default_loss_mask.copy()
        img = self.transform(img)

        data_dict = {
            'image': img,
            'loss_mask': loss_mask,
            'img_path': self.image_paths[idx]
        }

        if self.annos is not None:
            data_dict['label'] = self.annos[img_file]

        if self.stds is not None:
            data_dict['std'] = self.stds[img_file]
        else:
            data_dict['std'] = np.zeros(
                len(self.selected_defects), dtype=np.float32)

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
