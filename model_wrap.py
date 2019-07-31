from abc import ABC, abstractmethod
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import Loss, LossDict, separate_by_index, eval_mode
from metric import compute_spearman_rank_corr

__all__ = ['get_model_wrap']


def get_model_wrap(config, model, optimizer, device, knowledge_model=None):
    required_args = (model, optimizer, len(config.selected_defects),
                     config.sat_idx, device)
    if config.logits_act_type == 'sigmoid':
        BaseWrap = RegressionModel
        kwargs = {'sat_loss_weight': config.sat_loss_weight}
    elif config.logits_act_type == 'softmax':
        BaseWrap = SoftmaxClassificationModel
        kwargs = {'num_softmax_bins': config.num_softmax_bins,
                  'sat_num_softmax_bins': config.sat_num_softmax_bins}
    else:
        raise ValueError()

    if knowledge_model is None:
        wrap = BaseWrap(*required_args, **kwargs)
    else:
        wrap = get_knowledge_distill_wrap(
            BaseWrap, knowledge_model, config.kd_weight, *required_args, **kwargs)

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
        return self._outputs

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
        self._outputs = self.convert_logit_to_output(self.logits)

    def forward_and_compute_loss(self, data_batch):
        self.forward(data_batch)
        self.compute_loss()

    def train(self, data_batch):
        self.forward_and_compute_loss(data_batch)
        self.optimizer.zero_grad()
        self.loss_dict.reduce_all().backward_all()
        self.optimizer.step()

    def validate(self, dataloader, compute_spearman=True):
        score_dict, label_dict, loss_val_dict = self.gather(dataloader)

        # compute the mean loss values over the whole validation set
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
            scores = self.convert_output_to_score(self.outputs)
            scores = scores.detach().cpu().numpy()
            for n in range(scores.shape[0]):
                score_dict[self.img_paths[n]] = scores[n]

            # gather labels if provided
            if self.labels is not None:
                labels = self.labels.detach().cpu().numpy()
                for n in range(scores.shape[0]):
                    label_dict[self.img_paths[n]] = labels[n]

        return score_dict, label_dict, loss_val_dict


class RegressionModel(DefectDetection):
    def init(self, sat_loss_weight=None):
        self.loss_dict['Bce'] = Loss(
            nn.BCELoss(reduction='none'), reduce_func=self.reduce_func)
        self.loss_dict['Sat'] = Loss(
            nn.L1Loss(reduction='none'), reduce_func=self.reduce_func,
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

    def convert_output_to_score(self, outputs):
        return outputs


class SoftmaxClassificationModel(DefectDetection):
    @staticmethod
    def get_bins_interval(range_, num_bins):
        return range_ / (num_bins - 1)

    @staticmethod
    def convert_to_softmax_labels(labels, num_bins, sat_num_bins, sat_idx):
        interval = SoftmaxClassificationModel.get_bins_interval(1.0, num_bins)
        peak = torch.arange(0.05, 1.0 + 1e-3, interval)
        peak = torch.cat([torch.zeros(1), peak], dim=0).type_as(labels)

        sat_interval = SoftmaxClassificationModel.get_bins_interval(2.0, sat_num_bins)
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

    def init(self, num_softmax_bins=None, sat_num_softmax_bins=None):
        self.num_softmax_bins = num_softmax_bins
        self.sat_num_softmax_bins = sat_num_softmax_bins
        self.loss_dict['CE'] = Loss(
            nn.CrossEntropyLoss(reduction='none'), reduce_func=self.reduce_func)

    def convert_logit_to_output(self, logits):
        ''' Convert the logits to the outputs used for loss computation.
        Args:
            logits: the unnormalized outputs of the model.
        Returns:
            outputs: a list of length=self.num_outputs containing logits of shape
            [batch_size, self.sat_num_softmax_bins] for bad saturation and
            [batch_size, self.num_softmax_bins] for other defects.
        '''
        outputs = []
        num_pre_bins = 0
        for i in range(self.num_outputs):
            if i == self.sat_idx:
                bins = self.sat_num_softmax_bins
            else:
                bins = self.num_softmax_bins
            num_cur_bins = num_pre_bins + bins
            outputs.append(logits[:, num_pre_bins : num_cur_bins])
            num_pre_bins = num_cur_bins
        return outputs

    def compute_loss(self):
        softmax_labels = self.convert_to_softmax_labels(
            self.labels, self.num_softmax_bins, self.sat_num_softmax_bins, self.sat_idx)
        self.loss_dict['CE'].reset_state()
        for i in range(len(self.outputs)):
            self.loss_dict['CE'].accumulate(
                self.outputs[i], softmax_labels[:, i], self.loss_masks[:, i])

    def convert_output_to_score(self, outputs):
        ''' Convert the outputs used for loss computation to the final score.
        Args:
            outputs: the return values of method convert_logit_to_output.
        Returns:
            scores: the final scores of all defects.
        '''
        sat_outputs = F.softmax(outputs[self.sat_idx], dim=-1)
        outputs = outputs[:self.sat_idx] + outputs[self.sat_idx+1:]
        outputs = torch.stack(outputs, dim=0).transpose(0, 1)
        outputs = F.softmax(outputs, dim=-1)

        interval = self.get_bins_interval(1.0, self.num_softmax_bins)
        peak = torch.arange(0.0, 1.0 + 1e-3, interval)
        outputs = outputs * peak.type_as(outputs)
        outputs = outputs.sum(dim=-1)

        sat_interval = self.get_bins_interval(2.0, self.sat_num_softmax_bins)
        sat_peak = torch.arange(-1.0, 1.0 + 1e-3, sat_interval)
        sat_outputs = sat_outputs * sat_peak.type_as(sat_outputs)
        sat_outputs = sat_outputs.sum(dim=-1)

        scores = [outputs[:, :self.sat_idx],
                  sat_outputs.unsqueeze_(1), outputs[:, self.sat_idx:]]
        scores = torch.cat(scores, dim=1)

        return scores


# knowledge distilling class factory
def get_knowledge_distill_wrap(BaseWrap, knowledge_model, kd_weight,
                               *args, **kwargs):
    # set the knowledge model to inference only mode
    knowledge_model.eval()
    for p in knowledge_model.parameters():
        p.requires_grad = False

    # dynamic creation of class
    class_ = type('KnowledgeDistillModel', (BaseWrap,), {})

    # ------------------------------ Properties -------------------------------
    def fmaps_t(self):  # the feature maps of the training model
        return self._fmaps_t
    class_.fmaps_t = property(fmaps_t)

    def fmaps_k(self):  # the feature maps of the knowledge model
        return self._fmaps_k
    class_.fmaps_k = property(fmaps_k)
    # -------------------------------------------------------------------------

    # instantiate
    wrap = class_(*args, **kwargs)

    # add distillation loss
    wrap.loss_dict['KD'] = Loss(nn.MSELoss(), weight=kd_weight)

    # -------------------------- Overridden methods ---------------------------
    def forward(self, data_batch):
        super(wrap.__class__, self).forward(data_batch)
        self._fmaps_t = self.model.backbone.forward_kd(self.inputs[0])
        self._fmaps_k = knowledge_model.backbone.forward_kd(self.inputs[0])
    wrap.forward = types.MethodType(forward, wrap)

    def compute_loss(self):
        super(wrap.__class__, self).compute_loss()
        self.loss_dict['KD'].reset_state()
        for fmap_t, fmap_k in zip(self.fmaps_t, self.fmaps_k):
            self.loss_dict['KD'].accumulate(fmap_t, fmap_k)
    wrap.compute_loss = types.MethodType(compute_loss, wrap)
    # -------------------------------------------------------------------------

    return wrap
