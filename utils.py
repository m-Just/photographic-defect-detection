import csv
import functools
import glob
from collections.abc import Iterable
import json
import os
import random
import shutil
import sys
import time

from easydict import EasyDict as edict
import torch
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter


# ------------------------------- Helper classes -------------------------------

class Config(edict):
    def __init__(self, init=None):
        super(edict, self).__init__()
        if isinstance(init, str):
            self.load(init)
        elif isinstance(init, dict):
            self.update(init)
        else:
            raise TypeError()

    def clear(self):
        for k in self:
            delattr(self, k)
        super(edict, self).clear()

    def load(self, path, update_only=False):
        with open(path) as fp:
            loaded_config = json.load(fp)
        if not update_only:
            self.clear()
        self.update(loaded_config)
        print(f'Configurations loaded from {path}')
        return self

    def save(self, path):
        with open(path, 'w') as fp:
            json.dump(self, fp, indent=2)
        print(f'Configurations saved to {path}')

    def copy(self):
        return Config(init=self)


class Loss:
    def __init__(self, loss_func, weight=1.0, reduce_func=None):
        super(Loss, self).__init__()
        self.loss_func = loss_func
        self.weight = weight
        self.reduce_func = reduce_func
        self.reset_state()

    def __repr__(self): # a shorthand for printing readable loss value
        loss_val = self.reduce().data.cpu().numpy()
        return str(np.around(loss_val, decimals=4))

    def reset_state(self):
        self.reduced = self.reduce_func is None
        self.unweighted_value = None
        self.value = None

    def compute(self, y, t, loss_mask=None, **kwargs):
        self.reset_state()
        self.unweighted_value = self.loss_func(y, t, **kwargs)
        if loss_mask is not None:
            self.unweighted_value *= loss_mask
        self.value = self.weight * self.unweighted_value
        assert self.value.grad_fn
        return self.value

    def accumulate(self, y, t, loss_mask=None, **kwargs):
        loss_val = self.loss_func(y, t, **kwargs)
        if loss_mask is not None:
            loss_val *= loss_mask
        assert loss_val.grad_fn
        if self.unweighted_value is None:
            self.unweighted_value = loss_val
        else:
            self.unweighted_value += loss_val
        self.value = self.weight * self.unweighted_value
        assert self.value.grad_fn
        return self.value

    def reduce(self):
        if self.reduced:
            value = self.value
        else:
            unweighted_value = self.reduce_func(self.unweighted_value)
            value = self.weight * unweighted_value
        return value

    def reduce_(self):
        if not self.reduced:
            self.unweighted_value = self.reduce_func(self.unweighted_value)
            self.value = self.weight * self.unweighted_value
            self.reduced = True
        return self

    def backward(self, *args, **kwargs):
        self.value.backward(*args, **kwargs)


class LossDict(dict):
    def __init__(self):
        super(LossDict, self).__init__()

    def reduce_all(self):
        for name, loss in self.items():
            try:
                loss.reduce_()
            except Exception as e:
                print(f'Error when trying to reduce loss {name}')
                raise e
        return self

    def backward_all(self):
        total_loss = sum([l.value for l in self.values()])
        total_loss.backward()
        return self


class TensorboardWriter(SummaryWriter):
    def __init__(self, log_dir, defect_names, **kwargs):
        self.defect_names = defect_names
        if os.path.isdir(log_dir):
            shutil.rmtree(log_dir)
            time.sleep(5)   # wait for tensorboard to update
        super(TensorboardWriter, self).__init__(log_dir=log_dir, **kwargs)

    def add_dict(self, tag, dict_, num_steps):
        for name in dict_:
            val = dict_[name]
            self.add_scalar(f'{tag}/{name}', val, num_steps)

    def add_loss_dict(self, tag, loss_dict, num_steps):
        dict_ = {k: v.reduce() for k, v in loss_dict.items()}
        self.add_dict(tag, dict_, num_steps)

    def add_image_tensor(self, tag, image_tensor, num_steps):
        np_img = get_np_uint8_image(image_tensor)
        super(TensorboardWriter, self).add_image(tag, np_img, num_steps)

    def add_spearman_rank_corr(self, tag, corr_list, num_steps):
        names = ['Overall'] + self.defect_names
        self.add_dict(tag, {k: v for k, v in zip(names, corr_list)}, num_steps)


class CSV_Writer:
    def __init__(self, filepath, header, mode='a'):
        self.filepath = filepath
        self.num_cols = len(header)

        self.fileinit = not os.path.isfile(filepath)
        self.file = open(filepath, mode)
        self.writer = csv.writer(self.file)
        if self.fileinit or mode == 'w':
            self.writer.writerow(header)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def writerow(self, content):
        self.writer.writerow(content)

    def close(self):
        self.file.close()


class Timer:
    def __init__(self, name):
        self.name = name
        self.elapsed = 0.
        self._paused = True
        self._num_periods = 0

    @property
    def paused(self):
        return self._paused

    @property
    def num_periods(self):
        return self._num_periods

    @property
    def elapsed_average(self):
        if self.num_periods > 0:
            avg = self.elapsed / self.num_periods
        else:
            avg = 0.
        return avg

    def __enter__(self):
        self._paused = False
        self.start_time = time.time()
        return self

    def __exit__(self, type, value, traceback):
        self.pause()
        print(f'Time elapsed during {self.name}: {self.elapsed:.2f}s')

    def pause(self):
        if not self.paused:
            self.pause_time = time.time()
            self.elapsed += self.pause_time - self.start_time
            self._paused = True
            self._num_periods += 1
        else:
            raise RuntimeError()

    def resume(self):
        if self.paused:
            self.__enter__()
        else:
            raise RuntimeError()


# ------------------------------ Helper functions ------------------------------

def determine_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def at_interval(step, interval, start_index=1):
    return (step + start_index) % interval == 0


def separate_by_index(array, sat_idx):
    if isinstance(array, np.ndarray) or torch.is_tensor(array):
        r = list(range(sat_idx)) + list(range(sat_idx+1, array.shape[1]))
        return array[:, r], array[:, [sat_idx]]
    # elif isinstance(array, list):
    #     r = list(range(len(array)))
    #     r.remove(sat_idx)
    #     return [array[i] for i in r], [array[sat_idx]]
    elif isinstance(array, Iterable):
        raise NotImplementedError()
    else:
        raise TypeError()


def makedirs_if_not_exists(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir)


def get_np_uint8_image(img_tensor):
    return np.round(img_tensor.data.cpu().numpy() * 255).astype(np.uint8)


def select_by_indices(data, indices, dim, _current_dim=0):
    if isinstance(data, list):
        if dim == _current_dim:
            return [data[i] for i, d in enumerate(data) if i in indices]
        else:
            for i, d in enumerate(data):
                data[i] = select_by_indices(d, indices, dim, _current_dim + 1)
    elif isinstance(data, np.ndarray):
        if dim == _current_dim:
            return data[indices]
        else:
            for i, d in enumerate(data):
                data[i] = select_by_indices(d, indices, dim, _current_dim + 1)
    elif isinstance(data, dict):
        for name in data:
            data[name] = select_by_indices(data[name], indices, dim, _current_dim)
    else:
        raise NotImplementedError()
    return data


def get_images_recursively(root_path, shuffle=False):
    assert os.path.isdir(root_path), f'Path "{root_path}" does not exist'
    imgs = []
    for path in glob.glob(f'{root_path}/*'):
        if os.path.isdir(path):
            imgs += get_images_recursively(path)
        elif path.split('.')[-1].lower() in ['jpg', 'png', 'jpeg']:
            imgs.append(path)
    if shuffle:
        random.seed(423)
        random.shuffle(imgs)
    return imgs


def get_defect_names_by_idx(selected_defects=None):
    names = ['Bad Exposure', 'Bad White Balance', 'Bad Saturation', 'Noise',
             'Haze', 'Undesired Blur', 'Bad Composition']
    if selected_defects is not None:
        names = [name for i, name in enumerate(names) if i in selected_defects]
    return names


def ema_over_model_weights(ema_model, model, alpha):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def ema_over_state_dict(ema_state_dict, state_dict, alpha):
    for name in state_dict:
        new_state = state_dict[name].to(ema_state_dict[name].device)
        ema_state_dict[name].data.mul_(alpha).add_(1 - alpha, new_state.data)


def avg_over_state_dict(ckpt_paths):
    def load_ckpt(path):
        _state_dict = torch.load(path)
        print(f'Successfully loaded model {path}')
        return _state_dict

    ckpt_root, _ = os.path.split(ckpt_paths[0])
    avg_ckpt_path = f'{ckpt_root}/weight_averaged.pkl'
    if os.path.isfile(avg_ckpt_path):
        print('Found existing averaged model checkpoint')
        avg_state_dict = torch.load(avg_ckpt_path)
    else:
        avg_state_dict = load_ckpt(ckpt_paths[0])
        for ckpt_path in ckpt_paths[1:]:
            state_dict = load_ckpt(ckpt_path)
            for key in avg_state_dict.keys():
                avg_state_dict[key] += state_dict[key]

        for key in avg_state_dict.keys():
            avg_state_dict[key] /= len(ckpt_paths)

        torch.save(avg_state_dict, avg_ckpt_path)

    return avg_state_dict, avg_ckpt_path


# --------------------------------- Decorators ---------------------------------

def no_grad(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with torch.no_grad():
            result = func(*args, **kwargs)
        return result
    return wrapper


@no_grad
def eval_mode(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        model = args[0] if isinstance(args[0], nn.Module) else args[0].model
        training = model.training
        if training: model.eval()
        result = func(*args, **kwargs)
        if training: model.train()
        return result
    return wrapper


def config_override(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        new_args = []
        for i, arg in enumerate(args):
            if isinstance(arg, Config):
                arg = arg.copy()
                arg.update(kwargs)
            new_args.append(arg)
        return func(*new_args)
    return wrapper


def overrides(interface_class):
    @functools.wraps(interface_class)
    def overrider(method):
        assert(method.__name__ in dir(interface_class))
        return method
    return overrider


# --------------------------------- Debugging ---------------------------------

def trace_func_calls():
    def trace(frame, event, arg, indent=[0]):
        if event == "call":
          indent[0] += 2
          print("-" * indent[0] + "> call function", frame.f_code.co_name)
        elif event == "return":
          print("<" + "-" * indent[0], "exit function", frame.f_code.co_name)
          indent[0] -= 2
        return trace
    sys.settrace(trace)
