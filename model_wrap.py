from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import Network
from utils import Loss, LossDict, DataFlow
from utils import separate_by_index, eval_mode, overrides
from metric import compute_spearman_rank_corr
from dataloader import get_unlabeled_dataloader

__all__ = ['get_model_wrap']


def get_model_wrap(config, model, optimizer, status_dict, device):
    # determine the basic model wrap
    required_args = (model, optimizer, len(config.selected_defects),
                     config.sat_idx, device)
    if config.logits_act_type == 'sigmoid':
        Wrap = RegressionWrap
        kwargs = {'sat_loss_weight': config.sat_loss_weight}
    elif config.logits_act_type == 'softmax':
        Wrap = SoftmaxClassificationWrap
        kwargs = {'num_softmax_bins': config.num_softmax_bins,
                  'sat_num_softmax_bins': config.sat_num_softmax_bins}
    else:
        raise ValueError()

    # additional model wraps
    if config.use_kd:
        knowledge_model = Network(config, load_pretrained=True, ckpt_path='')
        knowledge_model = knowledge_model.to(device)

        # set the knowledge model to inference only mode
        knowledge_model.eval()
        for p in knowledge_model.parameters():
            p.requires_grad = False

        Wrap = get_knowledge_distill_wrap(
            Wrap, knowledge_model, config.kd_loss_weight)

    if config.use_mean_teacher:
        print('======== Creating teacher model ========')
        teacher_model = Network(config).to(device)  # TODO: copy directly
        teacher_model.train()
        for p in teacher_model.parameters():
            p.requires_grad = False
        print('========================================')
        unlabeled_dataloader = get_unlabeled_dataloader(config)
        unlabeled_dataflow = DataFlow(unlabeled_dataloader)
        Wrap = get_mean_teacher_wrap(
            Wrap, model, teacher_model, unlabeled_dataflow, config.cst_loss_weight,
            config.rampup_length, config.teacher_ema_alpha, status_dict)

    # DEBUG
    # print(Wrap.__mro__)

    wrap = Wrap(*required_args, **kwargs)

    return wrap


class DefectDetection(ABC):
    def __init__(self, model, optimizer, num_outputs, sat_idx, device,
                 **kwargs):
        self.model = model
        self.optimizer = optimizer
        self.num_outputs = num_outputs
        self.sat_idx = sat_idx
        self.device = device

        self._loss_dict = LossDict()

        self.init(**kwargs)

    # ------------------------------ Properties -------------------------------
    @property
    def inputs(self):
        return self._inputs

    @property
    def labels(self):
        return self._labels

    @property
    def loss_masks(self):
        return self._loss_masks

    @property
    def img_paths(self):
        return self._img_paths

    @property
    def logits(self):
        return self._logits

    @property
    def outputs(self):
        return self.convert_logit_to_output(self.logits)

    @property
    def scores(self):
        return self.convert_output_to_score(self.outputs)

    @property
    def loss_dict(self):
        return self._loss_dict

    # --------------------------- Abstract methods ----------------------------
    @abstractmethod
    def init(self, **kwargs):
        pass

    @abstractmethod
    def convert_logit_to_output(self, logits):
        pass

    @abstractmethod
    def compute_loss(self):
        pass

    @abstractmethod
    def convert_output_to_score(self, outputs):
        pass

    # ---------------------------- Static methods -----------------------------
    @staticmethod
    def reduce_func(x):
        return x.mean(0).sum()

    @staticmethod
    def allocate(data, name, device):
        d = data[name] if name in data else None
        if torch.is_tensor(d):
            d = d.to(device)
        return d

    # ---------------------------- Basic methods ------------------------------
    def parse_data(self, data):
        images = self.allocate(data, 'image', self.device)
        hybrid_ids = self.allocate(data, 'hybrid_id', self.device)
        self._inputs = (images, hybrid_ids)
        self._labels = self.allocate(data, 'label', self.device)
        self._loss_masks = self.allocate(data, 'loss_mask', self.device)
        self._img_paths = self.allocate(data, 'img_path', self.device)

    def forward(self, data_batch):
        self.parse_data(data_batch)
        self._logits = self.model(*self.inputs)

    def forward_and_compute_loss(self, data_batch):
        self.forward(data_batch)
        self.compute_loss()

    def train(self, data_batch):
        self.forward_and_compute_loss(data_batch)
        self.optimizer.zero_grad()
        self.loss_dict.reduce_all().backward_all()
        self.optimizer.step()

    def validate(self, dataloader, compute_loss=True, compute_spearman=True):
        score_dict, label_dict, loss_val_dict = self.gather(dataloader, gather_loss=compute_loss)

        # compute the mean loss values over the whole validation set
        if compute_loss:
            for loss_name, loss_vals in loss_val_dict.items():
                loss_val_dict[loss_name] = torch.mean(
                    torch.stack(loss_vals, dim=0))

        # compute the correlation
        corr = None
        if compute_spearman:
            corr = compute_spearman_rank_corr(
                score_dict, label_dict, self.num_outputs, self.sat_idx)

        return loss_val_dict, corr

    @eval_mode
    def gather(self, dataloader, gather_loss=True):
        assert not self.model.training

        score_dict = {}
        label_dict = {}
        loss_val_dict = {k: [] for k in self.loss_dict}

        for i, data_batch in enumerate(dataloader):
            # forward
            self.forward(data_batch)

            # gather loss values
            if gather_loss:
                if self.labels is None:
                    raise ValueError('No labels to compute the loss. '
                                     + 'To avoid the computation, '
                                     + 'you can set gather_loss=False')
                self.compute_loss()
                for loss_name in self.loss_dict:
                    reduced_val = self.loss_dict[loss_name].reduce().detach()
                    loss_val_dict[loss_name].append(reduced_val)

            # gather predicted scores
            scores = self.scores.detach().cpu().numpy()
            for n in range(scores.shape[0]):
                score_dict[self.img_paths[n]] = scores[n]

            # gather labels if provided
            if self.labels is not None:
                labels = self.labels.detach().cpu().numpy()
                for n in range(scores.shape[0]):
                    label_dict[self.img_paths[n]] = labels[n]

        return score_dict, label_dict, loss_val_dict


class RegressionWrap(DefectDetection):
    @property
    def stds(self):
        return self._stds

    @property
    def conf(self):
        return self._conf

    @overrides(DefectDetection)
    def parse_data(self, data):
        super(RegressionWrap, self).parse_data(data)
        self._stds = self.allocate(data, 'std', self.device)

    def compute_confidence(self):
        diff = (self.outputs.detach() - self.labels).abs()
        self._conf = torch.clamp(diff / (2 * self.stds + 1e-4), 0, 1)

    def init(self, sat_loss_weight=None):
        self.loss_dict['Bce'] = Loss(
            nn.BCELoss(reduction='none'), reduce_func=DefectDetection.reduce_func)
        self.loss_dict['Sat'] = Loss(
            nn.L1Loss(reduction='none'), reduce_func=DefectDetection.reduce_func,
            weight=sat_loss_weight)

    def convert_logit_to_output(self, logits):
        outputs = []
        for i in range(self.num_outputs):
            if i == self.sat_idx:
                outputs.append(logits[:, i])
            else:
                outputs.append(logits[:, i].sigmoid())
        return torch.stack(outputs, dim=1).type_as(logits)

    def compute_loss(self):
        outputs, sat_outputs = separate_by_index(self.outputs, self.sat_idx)
        labels, sat_labels = separate_by_index(self.labels, self.sat_idx)
        loss_masks, sat_loss_masks = separate_by_index(
            self.loss_masks, self.sat_idx)

        self.loss_dict['Bce'].compute(outputs, labels, loss_masks)
        self.loss_dict['Sat'].compute(sat_outputs, sat_labels, sat_loss_masks)

        if self.stds is not None:
            self.compute_confidence()
            conf, sat_conf = separate_by_index(self.conf, self.sat_idx)
            self.loss_dict['Bce'].apply(lambda v: v * conf)
            self.loss_dict['Sat'].apply(lambda v: v * sat_conf)

    def convert_output_to_score(self, outputs):
        return outputs


class SoftmaxClassificationWrap(DefectDetection):
    @staticmethod
    def get_bins_interval(range_, num_bins):
        return range_ / (num_bins - 1)

    @staticmethod
    def convert_to_softmax_labels(labels, num_bins, sat_num_bins, sat_idx):
        interval = SoftmaxClassificationWrap.get_bins_interval(1.0, num_bins)
        peak = torch.arange(0.05, 1.0 + 1e-3, interval)
        peak = torch.cat([torch.zeros(1), peak], dim=0).type_as(labels)

        sat_interval = SoftmaxClassificationWrap.get_bins_interval(2.0, sat_num_bins)
        peak_sat = torch.arange(-0.95, 0.95 + 1e-3, sat_interval)
        peak_sat = torch.cat([-torch.ones(1), peak_sat], dim=0).type_as(labels)

        labels_cv = []
        for label in labels:
            label_tmp = []
            for n, value in enumerate(label):
                peak_ = peak_sat if n == sat_idx else peak
                for i in range(len(peak_)):
                    if i < len(peak_) - 1:
                        if peak_[i] <= value < peak_[i+1]:
                            label_tmp.append(torch.Tensor([i]))
                    else:
                        if peak_[i] <= value:
                            label_tmp.append(torch.Tensor([i]))
            labels_cv.append(torch.stack(label_tmp, dim=1))
        labels_stack = torch.stack(labels_cv, dim=0)
        labels = torch.squeeze(labels_stack, dim=1).type_as(labels).long()
        return labels

    def _separate_bins(self, tensor):
        ''' Separate the flattened tensor by defects in order to compute the
            CrossEntropyLoss and convert it to scores.
        Args:
            tensor: tensor of shape [batch_size, total_num_softmax_bins].
        Returns:
            seps: a list of length=self.num_outputs containing tensors of shape
            [batch_size, self.sat_num_softmax_bins] for bad saturation and
            [batch_size, self.num_softmax_bins] for other defects.
        '''
        seps = []
        num_pre_bins = 0
        for i in range(self.num_outputs):
            if i == self.sat_idx:
                bins = self.sat_num_softmax_bins
            else:
                bins = self.num_softmax_bins
            num_cur_bins = num_pre_bins + bins
            seps.append(tensor[:, num_pre_bins : num_cur_bins])
            num_pre_bins = num_cur_bins
        return seps

    def init(self, num_softmax_bins=None, sat_num_softmax_bins=None):
        self.num_softmax_bins = num_softmax_bins
        self.sat_num_softmax_bins = sat_num_softmax_bins
        self.loss_dict['CE'] = Loss(
            nn.CrossEntropyLoss(reduction='none'), reduce_func=DefectDetection.reduce_func)

    def convert_logit_to_output(self, logits):
        bins = self._separate_bins(logits)

        bins, sat_bins = separate_by_index(bins, self.sat_idx)
        bins = torch.stack(bins, dim=0).transpose(0, 1)
        sat_bins = sat_bins[0]

        bins = F.softmax(bins, dim=-1)
        sat_bins = F.softmax(sat_bins, dim=-1)
        outputs = [bins[:, :self.sat_idx].flatten(start_dim=1),
                   sat_bins.flatten(start_dim=1),
                   bins[:, self.sat_idx:].flatten(start_dim=1)]
        outputs = torch.cat(outputs, dim=1)
        return outputs.type_as(logits)

    def compute_loss(self):
        softmax_labels = self.convert_to_softmax_labels(
            self.labels, self.num_softmax_bins, self.sat_num_softmax_bins, self.sat_idx)
        self.loss_dict['CE'].reset_state()
        softmax_bins = self._separate_bins(self.logits)
        for i in range(len(softmax_bins)):
            self.loss_dict['CE'].accumulate(
                softmax_bins[i], softmax_labels[:, i], self.loss_masks[:, i])

    def convert_output_to_score(self, outputs):
        ''' Convert the outputs used for loss computation to the final score.
        Args:
            outputs: the return values of method convert_logit_to_output.
        Returns:
            scores: the final scores of all defects.
        '''
        outputs = self._separate_bins(outputs)
        outputs, sat_outputs = separate_by_index(outputs, self.sat_idx)
        outputs = torch.stack(outputs, dim=0).transpose(0, 1)
        sat_outputs = sat_outputs[0]

        interval = self.get_bins_interval(1.0, self.num_softmax_bins)
        peak = torch.arange(0.0, 1.0 + 1e-3, interval)
        outputs = outputs * peak.type_as(self.logits)
        outputs = outputs.sum(dim=-1)

        sat_interval = self.get_bins_interval(2.0, self.sat_num_softmax_bins)
        sat_peak = torch.arange(-1.0, 1.0 + 1e-3, sat_interval)
        sat_outputs = sat_outputs * sat_peak.type_as(self.logits)
        sat_outputs = sat_outputs.sum(dim=-1)

        scores = [outputs[:, :self.sat_idx],
                  sat_outputs.unsqueeze_(1), outputs[:, self.sat_idx:]]
        scores = torch.cat(scores, dim=1)

        return scores


# knowledge distilling class factory
def get_knowledge_distill_wrap(BaseWrap, knowledge_model, kd_loss_weight):
    # dynamic creation of class
    class_ = type('KnowledgeDistillWrap', (BaseWrap,), {})

    # ------------------------------ Properties -------------------------------
    def fmaps_t(self):  # the feature maps of the training model
        return self._fmaps_t
    class_.fmaps_t = property(fmaps_t)

    def fmaps_k(self):  # the feature maps of the knowledge model
        return self._fmaps_k
    class_.fmaps_k = property(fmaps_k)

    # -------------------------- Overridden methods ---------------------------
    def init(self, **kwargs):
        super(class_, self).init(**kwargs)
        self.loss_dict['KD'] = Loss(nn.MSELoss(), weight=kd_loss_weight)
    class_.init = init

    def forward(self, data_batch):
        super(class_, self).forward(data_batch)
        self._fmaps_t = self.model.backbone.forward_kd(self.inputs[0])
        self._fmaps_k = knowledge_model.backbone.forward_kd(self.inputs[0])
    class_.forward = forward

    def compute_loss(self):
        super(class_, self).compute_loss()
        self.loss_dict['KD'].reset_state()
        for fmap_t, fmap_k in zip(self.fmaps_t, self.fmaps_k):
            self.loss_dict['KD'].accumulate(fmap_t, fmap_k)
    class_.compute_loss = compute_loss

    # NOTE: You can also replace a method on instance level at runtime by
    # wrap.compute_loss = types.MethodType(compute_loss, wrap)
    # -------------------------------------------------------------------------

    return class_


# mean teacher class factory
def get_mean_teacher_wrap(BaseWrap, student_model, teacher_model, unlabeled_dataflow,
                          cst_loss_weight, rampup_length, ema_alpha, status_dict):
    # dynamic creation of class
    class_ = type('MeanTeacherWrap', (BaseWrap,), {})

    # ------------------------------ Properties -------------------------------
    class_.rampup = property(lambda self: self._rampup)
    class_.weight_diff = property(lambda self: compute_model_difference())
    class_.unlabeled_inputs = property(lambda self: self._unlabeled_inputs)
    class_.unlabeled_loss_masks = property(lambda self: self._unlabeled_loss_masks)
    class_.unlabeled_img_paths = property(lambda self: self._unlabeled_img_paths)
    class_.unlabeled_logits = property(lambda self: self._unlabeled_logits)
    class_.unlabeled_outputs = property(lambda self: self.convert_logit_to_output(self.unlabeled_logits))
    class_.unlabeled_scores = property(lambda self: self.convert_output_to_score(self.unlabeled_outputs))
    class_.teacher_logits = property(lambda self: self._teacher_logits)
    class_.teacher_outputs = property(lambda self: self.convert_logit_to_output(self.teacher_logits))
    class_.teacher_scores = property(lambda self: self.convert_output_to_score(self.teacher_outputs))
    class_.unlabeled_teacher_logits = property(lambda self: self._unlabeled_teacher_logits)
    class_.unlabeled_teacher_outputs = property(lambda self: self.convert_logit_to_output(self.unlabeled_teacher_logits))
    class_.unlabeled_teacher_scores = property(lambda self: self.convert_output_to_score(self.unlabeled_teacher_outputs))

    # ---------------------------- Static methods -----------------------------
    def sigmoid_rampup(current):
        """Exponential rampup from https://arxiv.org/abs/1610.02242"""
        if rampup_length == 0:
            return 1.0
        else:
            current = np.clip(current, 0.0, rampup_length)
            phase = 1.0 - current / rampup_length
            return float(np.exp(-5.0 * phase * phase))

    def update_ema_variables(global_step):
        # Use the true average until the exponential average is more correct
        alpha = min(1 - 1 / (global_step + 1), ema_alpha)
        for ema_param, param in zip(teacher_model.parameters(), student_model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
        return alpha

    def compute_model_difference():
        param_num = 0
        param_sum = 0.
        teacher_params = dict(teacher_model.named_parameters())
        student_params = dict(student_model.named_parameters())
        for n in teacher_params:
            diff = teacher_params[n] - student_params[n]
            diff = diff.abs_().flatten()
            param_num += diff.size()[0]
            param_sum += diff.sum()
        return param_sum / param_num

    # -------------------------- Overridden methods ---------------------------
    def init(self, **kwargs):
        super(class_, self).init(**kwargs)
        self.loss_dict['Cst'] = Loss(nn.MSELoss(reduction='none'),
                                     reduce_func=DefectDetection.reduce_func,
                                     weight=cst_loss_weight)
        self.loss_dict['Cst_unlabeled'] = Loss(nn.MSELoss(reduction='none'),
                                               reduce_func=DefectDetection.reduce_func,
                                               weight=cst_loss_weight)
    class_.init = init

    def parse_data(self, data):
        super(class_, self).parse_data(data)
        unlabeled_data = unlabeled_dataflow.get_batch()
        images = self.allocate(unlabeled_data, 'image', self.device)
        hybrid_ids = self.allocate(unlabeled_data, 'hybrid_id', self.device)
        self._unlabeled_inputs = (images, hybrid_ids)
        self._unlabeled_loss_masks = self.allocate(unlabeled_data, 'loss_mask', self.device)
        self._unlabeled_img_paths = self.allocate(unlabeled_data, 'img_path', self.device)
    class_.parse_data = parse_data

    def forward(self, data_batch):
        super(class_, self).forward(data_batch)
        self._unlabeled_logits = student_model(*self.unlabeled_inputs)
        self._unlabeled_teacher_logits = teacher_model(*self.unlabeled_inputs)
        self._teacher_logits = teacher_model(*self.inputs)
    class_.forward = forward

    def compute_loss(self):
        super(class_, self).compute_loss()
        self.loss_dict['Cst'].compute(
            self.outputs, self.teacher_outputs)
        self.loss_dict['Cst_unlabeled'].compute(
            self.unlabeled_outputs, self.unlabeled_teacher_outputs) # TODO support loss masks

        # apply loss rampup
        self._rampup = sigmoid_rampup(status_dict['num_epoches'])
        rampup_func = lambda x: x * self._rampup
        self.loss_dict['Cst'].apply(rampup_func)
        self.loss_dict['Cst_unlabeled'].apply(rampup_func)
    class_.compute_loss = compute_loss

    def train(self, data_batch):
        super(class_, self).train(data_batch)
        update_ema_variables(status_dict['num_steps'])
    class_.train = train
    # -------------------------------------------------------------------------

    return class_
