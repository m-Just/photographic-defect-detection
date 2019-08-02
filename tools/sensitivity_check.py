'''This code snippet is copied from the old repository,
   thus you have to modify the code to solve path and dependency problems.
'''

import os

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter

from train import separate_by_index
from utils import makedirs_if_not_exists
from eval import get_test_dataloader, restore_model

from global_constants import DEFECT_NAMES


def score_convert_reg(outputs, selected_defects, sat_idx):
    outputs, sat_outputs = separate_by_index(outputs, selected_defects, sat_idx)
    outputs = torch.sigmoid(outputs)[0]
    return torch.cat([outputs[:2], sat_outputs[0], outputs[2:]], dim=0)


def score_convert_softmax_cls(outputs, selected_defects, sat_idx):
    # outputs is a vector of length 11*2 + 21 + 11*4 = 87,
    # which corresponds to [11, 11, 21, 11, 11, 11, 11] for the 7 defects.
    softmax_input_num = [11, 11, 21, 11, 11, 11, 11]
    softmax_input_num = [softmax_input_num[n] for n in selected_defects]

    # we split the output by the defects
    result = []
    sat_result = []
    n_pre = 0
    for n in range(len(selected_defects)):
        if n == sat_idx:    # if this is for saturation
            sat_result.append(outputs[n_pre:n_pre + softmax_input_num[n]].data.cpu().numpy())
            n_pre += softmax_input_num[n]
        else:
            result.append(outputs[n_pre:n_pre + softmax_input_num[n]].data.cpu().numpy())
            n_pre += softmax_input_num[n]

    # now result is a nested list containing 6 lists of length 11,
    # and sat_result is a nested list containing a single list of length 11.
    # stack up the result into a matrix -> result.shape=(6, 11)
    result = np.stack(result, axis=0)

    # also put the saturation result into a matrix -> result.shape=(1, 21)
    sat_result = np.stack(sat_result, axis=0)

    # convert them to torch tensors for convenient softmax computation
    # note: this does not change any values of matrices
    result = torch.FloatTensor(result)
    sat_result = torch.FloatTensor(sat_result)

    # compute the softmax of the raw scores along dimension 1.
    # note: result.shape does not change.
    result = F.softmax(result, dim=1)
    sat_result = F.softmax(sat_result, dim=1)

    # define constant values for computing the final score.
    # the first one is an array:
    # [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0].
    peak = torch.range(0, 1, 0.1)

    # the second one is also an array, with additional negative values:
    # [-1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0,
    #   0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  1.0].
    sat_peak = torch.range(-1, 1, 0.1)

    # compute the element-wise product between softmax results and peaks.
    # note: result.shape does not change.
    result = result * peak
    sat_result = sat_result * sat_peak

    # sum up on dimension 1, and now result.shape=(6,) and sat_result.shape=(1,)
    result = result.sum(dim=1)
    sat_result = sat_result.sum(dim=1)

    # rearrange them to match with the final output order:
    # vector = [result[0], result[1], sat_result[0], result[2], ..., result[5]],
    # of which the length is 7.
    return torch.cat([result[:sat_idx], sat_result, result[sat_idx:]], dim=0)


def get_range(min_val, max_val, num_intervals=10, interval=None, mode='linear'):
    '''
    Get an inclusive range for both min_val and max_val,
    `num_intervals' will be overridden if `interval' is provided.
    '''
    if interval is not None:
        num_intervals = int(float(max_val - min_val) / interval)

    if mode == 'linear':
        _range = list(np.linspace(min_val, max_val, num=num_intervals))
    elif mode == 'log':
        _range = list(np.logspace(min_val, max_val, num=num_intervals, base=2))
    else:
        raise NotImplementedError()

    return _range


def pil_to_numpy(image):
    return np.asarray(image, dtype=np.float32)


def process_in_numpy(func):
    def wrapper(*args, **kwargs):
        args = list(args)
        args[0] = pil_to_numpy(args[0])
        np_image = func(*args, **kwargs)
        np_image = np.clip(np_image, 0., 255.)
        return Image.fromarray(np.round(np_image).astype(np.uint8))
    return wrapper


@process_in_numpy
def adjust_white_balance(image, mult=[1.0, 1.0, 1.0]):
    return image * np.array(mult)


@process_in_numpy
def apply_gaussian_noise(image, sigma, mult=10.):
    return image + np.random.normal(scale=sigma, size=image.shape) * mult


@process_in_numpy
def apply_gaussian_blur(image, sigma):
    return gaussian_filter(image, sigma=[sigma, sigma, 0])


@process_in_numpy
def apply_haze(image, factor):
    return image + factor * 255.


@process_in_numpy
def center_collage(image, image_aug, scale):
    h, w, c = image.shape
    h_crop = int(h * scale)
    w_crop = int(w * scale)
    h_offset = (h - h_crop) // 2
    w_offset = (w - w_crop) // 2
    np_image_aug = pil_to_numpy(image_aug)
    image[h_offset : h_offset + h_crop, w_offset : w_offset + w_crop] = \
        np_image_aug[h_offset : h_offset + h_crop, w_offset : w_offset + w_crop]
    return image


def predict(ckpt_root, selected_defects, aug_params, model_kwargs, name_dict):
    # determine training device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load images
    test_loader = get_test_dataloader('sensitivity/test_imgs', transforms.ToTensor(), test_batch_size=1)

    # prepare model
    model = restore_model(ckpt_root, 100, selected_defects, 'shufflenet', 'eval', device,
                          **model_kwargs)

    results = dict()
    for i, data in enumerate(test_loader):
        # load image
        images = data['image']
        image_ids = data['img_id']
        assert images.size()[0] == 1
        image_pil = TF.to_pil_image(images[0])
        image_id = image_ids[0]

        for name in augs:
            if name not in results:
                results[name] = [[] for _ in range(len(augs[name]['params']))]
            for k, param in enumerate(augs[name]['params']):
                print('image #%02d, %s, %.3f' % (i, name, param))
                # apply augmentation to the image
                image = augs[name]['method'](image_pil, param)

                # convert to tensor and resize
                image = TF.to_tensor(image)
                image = image.unsqueeze_(0).to(device)
                image = F.interpolate(image, size=(224, 224), mode='area')

                # save the augmented image
                save_dir = 'sensitivity/%s' % name
                img_savepath = '%s/%s_%.3f.png' % (
                    save_dir, image_id.split('/')[-1].split('.')[0], param)
                if not os.path.isfile(img_savepath):
                    makedirs_if_not_exists(save_dir)
                    image_ = TF.to_pil_image(image.data.cpu()[0])
                    image_.save(img_savepath)

                # gather model prediction
                outputs = model(image)
                if model_kwargs['use_softmax_classifier']:
                    outputs = torch.cat([outputs[0][:2].flatten(), outputs[1].flatten(), outputs[0][2:].flatten()], dim=0)
                    outputs = score_convert_softmax_cls(outputs, selected_defects, 2)
                else:
                    outputs = score_convert_reg(outputs, selected_defects, 2)
                outputs = outputs.data.cpu().numpy()
                print(outputs)
                idx = DEFECT_NAMES.index(name_dict[name])
                idx = selected_defects.index(idx)
                results[name][k].append(outputs[idx])

    # plot graph
    for name in augs:
        r = np.array(results[name])
        r = np.mean(r, axis=1)
        fig, ax = plt.subplots()
        ax.plot(augs[name]['params'], r)
        ax.set(xlabel='factor', ylabel='score',
               title=name)
        ax.grid()
        fig.savefig('sensitivity/%s.png' % name)
        plt.clf()

if __name__ == '__main__':
    # 0.2.5
    # ckpt_root = 'ckpts/baseline_trunc_stage_v2_balance_setting2'
    # model_kwargs = {}

    # 0.2.8b
    ckpt_root = 'ckpts/baseline_trunc_stage_v2_hybrid3_v2_softmax_classifier'
    model_kwargs = {'use_softmax_classifier': True,
                    'hybrid': 3}

    selected_defects = [0, 1, 2, 3, 4, 5, 6]
    augs = {
        'Bad Exposure': {
            'method': TF.adjust_gamma,
            'params': get_range(-2.0, 2.0, num_intervals=20, mode='log')
        },
        'Bad White Balance': {
            'method': lambda img, b: adjust_white_balance(img, mult=[1.0, 1.0, b]),
            'params': get_range(0.0, 2.0, num_intervals=20)
        },
        'Bad Saturation': {
            'method': TF.adjust_saturation,
            'params': get_range(0.0, 2.0, num_intervals=20)
        },
        'Noise': {
            'method': apply_gaussian_noise,
            'params': get_range(0.0, 8.0, num_intervals=20)
        },
        'Haze': {
            'method': apply_haze,
            'params': get_range(0.0, 0.5, num_intervals=20)
        },
        'Undesired Blur(sigma)': {
            'method': apply_gaussian_blur,
            'params': get_range(0.0, 4.0, num_intervals=20)
        },
        'Undesired Blur(region)': {
            'method': lambda img, scale: center_collage(img, apply_gaussian_blur(img, 2.0), scale),
            'params': get_range(0.0, 1.0, num_intervals=20)
        }
    }

    name_dict = {
        'Bad Exposure': 'Bad Exposure',
        'Bad White Balance': 'Bad White Balance',
        'Bad Saturation': 'Bad Saturation',
        'Noise': 'Noise',
        'Haze': 'Haze',
        'Undesired Blur(sigma)': 'Undesired Blur',
        'Undesired Blur(region)': 'Undesired Blur'
    }

    predict(ckpt_root, selected_defects, augs, model_kwargs, name_dict)
