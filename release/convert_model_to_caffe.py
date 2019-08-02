from __future__ import print_function
import os
import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import caffe
from caffe import layers as L
from caffe import params as P
from caffe.model_libs import ConvBNLayer
import cv2
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, 'models')
# sys.path.insert(0, '/home/SENSETIME/likaican/Projects/server/defect-detection-refactor/models')
from ShuffleNet import ShuffleNet_Multi_Regression as ShuffleNet_Multi_Regression_Pytorch
from ShuffleNet_wang import ShuffleNet_Multi_SoftmaxClassifiers as ShuffleNet_Multi_SoftmaxClassifiers_Pytorch
from preprocess import preprocess_v2, convert_to_tensor

def conv3x3(net, from_layer, output_layer,
            in_channels, out_channels, use_bn=False, stride=1,
            padding=1, bias=True, groups=1):
    return ConvBNLayer(net, from_layer, output_layer, use_bn=use_bn, act_f=None,
                       conv_f='conv', num_output=out_channels,
                       kernel_size=3, pad=padding, stride=stride,
                       use_scale=use_bn, lr_mult=0.0, group=groups,
                       use_global_stats=True)

def conv1x1(net, from_layer, output_layer, use_bn, act_f,
            in_channels, out_channels, groups=1):
    return ConvBNLayer(net, from_layer, output_layer, use_bn=use_bn, act_f=act_f,
                       conv_f='conv', num_output=out_channels,
                       kernel_size=1, pad=0, stride=1,
                       use_scale=use_bn, lr_mult=0.0, group=groups,
                       use_global_stats=True)

class Classifier():
    def __init__(self, net, from_layer, id, num_inputs, num_classifier=4):
        self.net = net
        result_list = []
        for i in range(num_classifier):
            self.net['cls{}_{}'.format(id, i)] = L.InnerProduct(self.net[from_layer], num_output=1)
            self.net['sigmoid{}_{}'.format(id, i)] = L.Sigmoid(self.net['cls{}_{}'.format(id, i)], in_place=False)
            result_list.append(self.net['sigmoid{}_{}'.format(id, i)])
        self.net['cls{}_sum'.format(id)] = L.Eltwise(*result_list)

# class SoftmaxClassifier():
#     def __init__(self, net, from_layer, id, num_inputs, num_class):
#         self.net = net
#         self.net['hybrid_classifiers_{}'.format(id)] = L.InnerProduct(self.net[from_layer], num_output=num_class)

class ShuffleUnit():
    def __init__(self, net, name, from_layer, in_channels, out_channels, groups=3,
                 grouped_conv=True, combine='add', use_bn=True):

        self.net = net
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grouped_conv = grouped_conv
        self.combine = combine
        self.groups = groups
        self.bottleneck_channels = self.out_channels // 4

        # define the type of ShuffleUnit
        if self.combine == 'add':
            # ShuffleUnit Figure 2b
            self.depthwise_stride = 1
        elif self.combine == 'concat':
            # ShuffleUnit Figure 2c
            self.depthwise_stride = 2

            # ensure output of concat has the same channels as
            # original output channels.
            self.out_channels -= self.in_channels
        else:
            raise ValueError("Cannot combine tensors with \"{}\"" \
                             "Only \"add\" and \"concat\" are" \
                             "supported".format(self.combine))

        # Use a 1x1 grouped or non-grouped convolution to reduce input channels
        # to bottleneck channels, as in a ResNet bottleneck module.
        # NOTE: Do not use group convolution for the first conv1x1 in Stage 2.
        self.first_1x1_groups = self.groups if grouped_conv else 1

        residual = from_layer
        if self.combine == 'concat':
            self.net[name + '_avgpool'] = \
                L.Pooling(self.net[residual], pool=P.Pooling.AVE, kernel_size=3,
                          stride=2, pad=1)
            residual = name + '_avgpool'

        self.g_conv_1x1_compress = self._make_grouped_conv1x1(
            from_layer = from_layer,
            output_layer = name + '_1x1_compress',
            in_channels = self.in_channels,
            out_channels = self.bottleneck_channels,
            groups = self.first_1x1_groups,
            batch_norm = use_bn,
            relu = True
        )
        from_layer = name + '_1x1_compress_relu'

        self.net[name + '_channel_shuffle'] =\
            L.ShuffleChannel(self.net[from_layer], shuffle_channel_param=dict(group=groups))
        from_layer = name + '_channel_shuffle'

        # 3x3 depthwise convolution followed by batch normalization
        self.depthwise_conv3x3 = conv3x3(
            net = self.net,
            from_layer = from_layer,
            output_layer = name + '_depthwise_3x3',
            in_channels = self.bottleneck_channels,
            out_channels = self.bottleneck_channels,
            use_bn = use_bn,
            stride = self.depthwise_stride,
            groups = self.bottleneck_channels
        )
        from_layer = name + '_depthwise_3x3'
        if use_bn:
            from_layer = from_layer + '_scale'

        # Use 1x1 grouped convolution to expand from
        # bottleneck_channels to out_channels
        self.g_conv_1x1_expand = self._make_grouped_conv1x1(
            from_layer = from_layer,
            output_layer = name + '_1x1_expand',
            in_channels = self.bottleneck_channels,
            out_channels = self.out_channels,
            groups = self.groups,
            batch_norm = use_bn,
            relu = False
        )
        from_layer = name + '_1x1_expand'
        if use_bn:
            from_layer = from_layer + '_scale'

        if combine == 'add':
            self.net[name + '_add'] = L.Eltwise(self.net[residual],
                                                self.net[from_layer])
            from_layer = name + '_add'

        elif combine == 'concat':
            self.net[name + '_concat'] = L.Concat(self.net[residual],
                                                  self.net[from_layer],
                                                  axis=1)
            from_layer = name + '_concat'

        self.net[name + '_final_relu'] = L.ReLU(self.net[from_layer], in_place=False)
        self.out_layer = name + '_final_relu'

    def _make_grouped_conv1x1(self, from_layer, output_layer,
        in_channels, out_channels, groups,
        batch_norm=True, relu=False):

        act_f = 'relu' if relu else None

        conv = conv1x1(self.net, from_layer, output_layer, use_bn=batch_norm,
                       act_f=act_f, in_channels=in_channels,
                       out_channels=out_channels, groups=groups)

        return conv

class ShuffleNet_Multi_Regression():
    """ShuffleNet for multiple regression."""

    def __init__(self, net, from_layer, groups=3, in_channels=3, num_output=7,
                 trunc_stage=False, global_pooling='combined', use_bn=True,
                 hybrid=0, sid=None):
        """ShuffleNet constructor.

        Arguments:
            groups (int, optional): number of groups to be used in grouped
                1x1 convolutions in each ShuffleUnit. Default is 3 for best
                performance according to original paper.
            in_channels (int, optional): number of channels in the input tensor.
                Default is 3 for RGB image inputs.
            num_output (int, optional): number of classes to predict. Default
                is 1000 for ImageNet.

        """
        self.net = net
        self.groups = groups
        self.stage_repeats = [3, 7, 3]
        self.in_channels = in_channels
        self.trunc_stage = trunc_stage
        self.global_pooling = global_pooling
        self.use_bn = use_bn

        # index 0 is invalid and should never be called.
        # only used for indexing convenience.
        if groups == 1:
            self.stage_out_channels = [-1, 24, 144, 288, 567]
        elif groups == 2:
            self.stage_out_channels = [-1, 24, 200, 400, 800]
        elif groups == 3:
            self.stage_out_channels = [-1, 24, 240, 480, 960]
        elif groups == 4:
            self.stage_out_channels = [-1, 24, 272, 544, 1088]
        elif groups == 8:
            self.stage_out_channels = [-1, 24, 384, 768, 1536]
        else:
            raise ValueError(
                """{} groups is not supported for
                   1x1 Grouped Convolutions""".format(groups))

        # Stage 1 always has 24 output channels
        self.conv1 = conv3x3(self.net, from_layer, 'stage1_conv1',
                    self.in_channels, self.stage_out_channels[1], stride=2)
        from_layer = 'stage1_conv1'
        self.net['stage1_maxpool'] = L.Pooling(self.net[from_layer],
            pool=P.Pooling.MAX, kernel_size=3, stride=2, pad=1)
        from_layer = 'stage1_maxpool'

        # print(self.net['stage1_conv1'])
        # exit()

        # Stage 2
        self.stage2 = self._make_stage(from_layer, 2)
        if self.trunc_stage:
            self.stage_repeats = [3, 3]
            # Stage 3
            self.stage3 = self._make_stage(self.stage2, 3)
            from_layer = self.stage3
        else:
            # Stage 3
            self.stage3 = self._make_stage(self.stage2, 3)
            # Stage 4
            self.stage4 = self._make_stage(self.stage3, 4)
            from_layer = self.stage4

        # Global pooling:
        if self.global_pooling == 'combined':
            self.net['avg_pooling'] = \
                L.Pooling(self.net[from_layer], pool=P.Pooling.AVE, global_pooling=True)
            self.net['max_pooling'] = \
                L.Pooling(self.net[from_layer], pool=P.Pooling.MAX, global_pooling=True)
            self.net['global_pooling'] = L.Eltwise(self.net['max_pooling'] / 2, self.net['avg_pooling'] / 2)
        else:
            self.net['global_pooling'] = \
                L.Pooling(self.net[from_layer], pool=P.Pooling.AVE, global_pooling=True)
        from_layer = 'global_pooling'

        # Fully-connected classification layer
        self.net['flatten'] = L.Flatten(self.net[from_layer], axis=1)
        from_layer = 'flatten'

        result_list = []
        if hybrid:
            self.net['fc_{}'.format(sid)] = L.InnerProduct(self.net[from_layer], num_output=num_output)
            slices = L.Slice(self.net['fc_{}'.format(sid)], ntop=num_output,
                                        slice_dim=1, slice_point=list(range(1, num_output)))

            for i in range(num_output):
                slice = self.net['slice_{}'.format(i)] = slices[i]
                if i == 2:  # saturation branch
                    result_list.append(slice)
                else:
                    self.net['sigmoid_{}'.format(i)] = L.Sigmoid(slice, in_place=False)
                    result_list.append(self.net['sigmoid_{}'.format(i)])
        else:
            for i in range(num_output):
                # self.net['dropout_{}'.format(i)] = L.Dropout(self.net[from_layer], dropout_ratio=0.5, in_place=False)
                # from_layer = 'dropout_{}'.format(i)
                self.net['fc_{}'.format(i)] = L.InnerProduct(self.net[from_layer], num_output=1)
                if i == 2:  # saturation branch
                    result_list.append(self.net['fc_{}'.format(i)])
                else:
                    self.net['sigmoid_{}'.format(i)] = L.Sigmoid(self.net['fc_{}'.format(i)], in_place=False)
                    result_list.append(self.net['sigmoid_{}'.format(i)])

        self.net['final_concat'] = L.Concat(*result_list, axis=1)
        self.out_layer = self.net['final_concat']
        from_layer = 'final_concat'

    def _make_stage(self, from_layer, stage):

        stage_name = "stage{}".format(stage)

        # First ShuffleUnit in the stage
        # 1. non-grouped 1x1 convolution (i.e. pointwise convolution)
        #   is used in Stage 2. Group convolutions used everywhere else.
        grouped_conv = stage > 2

        # 2. concatenation unit is always used.
        first_module = ShuffleUnit(self.net, stage_name + '_0', from_layer,
                                   self.stage_out_channels[stage-1],
                                   self.stage_out_channels[stage], groups=self.groups,
                                   grouped_conv=grouped_conv, combine='concat',
                                   use_bn=self.use_bn)
        self.net = first_module.net
        from_layer = first_module.out_layer

        # add more ShuffleUnits depending on pre-defined number of repeats
        for i in range(self.stage_repeats[stage-2]):
            name = stage_name + "_{}".format(i+1)
            module = ShuffleUnit(
                self.net, name, from_layer,
                self.stage_out_channels[stage],
                self.stage_out_channels[stage],
                groups=self.groups,
                grouped_conv=True,
                combine='add',
                use_bn=self.use_bn
            )
            self.net = module.net
            from_layer = module.out_layer

        return from_layer

class ShuffleNet_Multi_Classifiers():
    def __init__(self, net, from_layer, groups=8, in_channels=3, output_num=7,
                 global_pooling_mode='average', cnum_ratio='1,1',
                 trunc_stage=False, use_bn=True, logits_num=None, hybrid=0):
        self.net = net
        self.groups = groups
        self.stage_repeats = [3, 7, 3]
        self.in_channels = in_channels
        self.output_num = output_num
        self.global_pooling_mode = global_pooling_mode
        self.trunc_stage = trunc_stage
        self.use_bn = use_bn
        self.logits_num = logits_num
        self.hybrid = hybrid

        if self.trunc_stage:
            self.stage_repeats = [3, 3]

        # index 0 is invalid and should never be called.
        # only used for indexing convenience.
        if groups == 1:
            self.stage_out_channels = [-1, 24, 144, 288, 567]
        elif groups == 2:
            self.stage_out_channels = [-1, 24, 200, 400, 800]
        elif groups == 3:
            self.stage_out_channels = [-1, 24, 240, 480, 960]
        elif groups == 4:
            self.stage_out_channels = [-1, 24, 272, 544, 1088]
        elif groups == 8:
            self.stage_out_channels = [-1, 24, 384, 768, 1536]
        else:
            raise ValueError(
                """{} groups is not supported for
                   1x1 Grouped Convolutions""".format(groups))

        # Stage 1 always has 24 output channels
        self.conv1 = conv3x3(self.net, from_layer, 'stage1_conv1',
                    self.in_channels, self.stage_out_channels[1], stride=2)
        from_layer = 'stage1_conv1'
        self.net['stage1_maxpool'] = L.Pooling(self.net[from_layer],
            pool=P.Pooling.MAX, kernel_size=3, stride=2, pad=1)
        from_layer = 'stage1_maxpool'

        # Stage 2
        self.stage2 = self._make_stage(from_layer, 2)
        # Stage 3
        self.stage3 = self._make_stage(self.stage2, 3)
        # Stage 4
        if self.trunc_stage:
            from_layer = self.stage3
        else:
            self.stage4 = self._make_stage(self.stage3, 4)
            from_layer = self.stage4

        # Global pooling:
        if self.global_pooling_mode == 'combined':
            self.net['avg_pooling'] = \
                L.Pooling(self.net[from_layer], pool=P.Pooling.AVE, global_pooling=True)
            self.net['max_pooling'] = \
                L.Pooling(self.net[from_layer], pool=P.Pooling.MAX, global_pooling=True)
            self.net['global_pooling'] = L.Eltwise(self.net['max_pooling'] / 2, self.net['avg_pooling'] / 2)
        else:
            self.net['global_pooling'] = \
                L.Pooling(self.net[from_layer], pool=P.Pooling.AVE, global_pooling=True)
        from_layer = 'global_pooling'

        # Fully-connected classification layer
        if self.trunc_stage:
            num_inputs = self.stage_out_channels[-2]
        else:
            num_inputs = self.stage_out_channels[-1]

        self.net['flatten'] = L.Flatten(self.net[from_layer], axis=1)
        from_layer = 'flatten'

        result_list = []
        if hybrid:
            classifier_list = []
            for h in range(self.hybrid):
                self.net['hybrid_classifiers_{}'.format(h)] = L.InnerProduct(self.net[from_layer], num_output=sum(logits_num))
                self.net['hybrid_unsqueeze_{}'.format(h)] = L.Reshape(self.net['hybrid_classifiers_{}'.format(h)], reshape_param={'shape':{'dim': [0, 1, -1]}})
                classifier_list.append(self.net['hybrid_unsqueeze_{}'.format(h)])
                # classifier_list.append(self.net['hybrid_unsqueeze_{}'.format(h)])

            # DEBUG
            # self.net['hybrid_sum'] = L.Eltwise(*classifier_list)
            self.net['hybrid_concat'] = L.Concat(*classifier_list, axis=1)
            # self.net['hybrid_mean'] = L.Reduction(self.net['hybrid_concat'], axis=2)#, reduction_param={'operation': 4, 'axis': 1})   # reduce_mean
        else:
            for i in range(self.output_num-1):
                self.net = Classifier(self.net, from_layer, i, num_inputs).net
                result_list.append(self.net['cls{}_sum'.format(i)])
            self.net['cls_flatten'] = L.Concat(*result_list, axis=1)
            self.net['fc'] = L.InnerProduct(self.net[from_layer], num_output=1)
            self.net['final_concat'] = L.Concat(self.net['cls_flatten'], self.net['fc'], axis=1)
            self.out_layer = self.net['final_concat']
        from_layer = 'final_concat'

    def _make_stage(self, from_layer, stage):

        stage_name = "stage{}".format(stage)

        # First ShuffleUnit in the stage
        # 1. non-grouped 1x1 convolution (i.e. pointwise convolution)
        #   is used in Stage 2. Group convolutions used everywhere else.
        grouped_conv = stage > 2

        # 2. concatenation unit is always used.
        first_module = ShuffleUnit(self.net, stage_name + '_0', from_layer,
                                   self.stage_out_channels[stage-1],
                                   self.stage_out_channels[stage], groups=self.groups,
                                   grouped_conv=grouped_conv, combine='concat',
                                   use_bn=self.use_bn)
        self.net = first_module.net
        from_layer = first_module.out_layer

        # add more ShuffleUnits depending on pre-defined number of repeats
        for i in range(self.stage_repeats[stage-2]):
            name = stage_name + "_{}".format(i+1)
            module = ShuffleUnit(
                self.net, name, from_layer,
                self.stage_out_channels[stage],
                self.stage_out_channels[stage],
                groups=self.groups,
                grouped_conv=True,
                combine='add',
                use_bn=self.use_bn
            )
            self.net = module.net
            from_layer = module.out_layer

        return from_layer

def gen_network():
    net = caffe.NetSpec()
    from_layer = 'data'
    shape = [1, 3, INPUT_SIZE, INPUT_SIZE]
    net.data = L.Input(shape=dict(dim=shape))

    if HEAD_TYPE == 'reg':
        net = ShuffleNet_Multi_Regression(net, from_layer,
                                          num_output=OUTPUT_NUM,
                                          trunc_stage=TRUNC_STAGE,
                                          global_pooling=GLOBAL_POOLING,
                                          groups=GROUPS,
                                          use_bn=USE_BN,
                                          hybrid=HYBRID,
                                          sid=SID).net
    elif HEAD_TYPE == 'cls':
        net = ShuffleNet_Multi_Classifiers(net, from_layer, groups=GROUPS,
                                          global_pooling_mode=GLOBAL_POOLING,
                                          trunc_stage=TRUNC_STAGE,
                                          use_bn=USE_BN,
                                          logits_num=LOGITS_NUM,
                                          hybrid=HYBRID).net
    return net

def genAndwrite(prototxt_path, model_name):
    Net = gen_network()
    train_net_file = "{}".format(prototxt_path)
    with open(train_net_file, 'w') as f:
        print('name: "{}_train"'.format(model_name), file=f)
        print(Net.to_proto(), file=f)

def gen_caffemodel(caffe_proto, ckpt_path, save_path):
    net_caffe = caffe.Net(caffe_proto,caffe.TEST)
    pytorch_model = torch.load(ckpt_path, map_location='cpu')
    pytorch_item_key_list = [k for k,v in pytorch_model.items()]
    cnt = 0
    for k,v in net_caffe.params.items():
        print(k)
        if 'bn' in k:
            pytorch_key = pytorch_item_key_list[cnt+2]
            print(pytorch_key)
            v[0].data[::] = pytorch_model[pytorch_key].cpu().numpy()
            pytorch_key = pytorch_item_key_list[cnt+3]
            print(pytorch_key)
            v[1].data[::] = pytorch_model[pytorch_key].cpu().numpy()
            v[2].data[::] = 1.0
            # pass
        elif 'scale' in k:
            pytorch_key = pytorch_item_key_list[cnt]
            print(pytorch_key)
            v[0].data[::] = pytorch_model[pytorch_key].cpu().numpy()
            pytorch_key = pytorch_item_key_list[cnt+1]
            print(pytorch_key)
            v[1].data[::] = pytorch_model[pytorch_key].cpu().numpy()
            cnt = cnt + 5
        elif 'upsample' in k:
            continue
        elif 'up_bilinear' in k:
            pass
        else:
            pytorch_key = pytorch_item_key_list[cnt]
            print(pytorch_key)
            v[0].data[::] = pytorch_model[pytorch_key].cpu().numpy()
            cnt = cnt + 1
            pytorch_key = pytorch_item_key_list[cnt]
            print(pytorch_key)
            v[1].data[::] = pytorch_model[pytorch_key].cpu().numpy()
            cnt = cnt + 1
    net_caffe.save(save_path)

def load_image(img_path):
    test_img = cv2.imread(img_path)[:, :, ::-1].astype(np.uint8)
    test_img = cv2.resize(test_img, (RESIZE, RESIZE), interpolation=cv2.INTER_AREA)
    delta = (RESIZE - INPUT_SIZE) // 2
    test_img = test_img[delta : RESIZE - delta, delta : RESIZE - delta, :]
    print(test_img.shape)
    test_img = test_img.astype(np.float32) / 255.0
    test_img = np.transpose(test_img, (2, 0, 1))[np.newaxis, ...]
    return test_img

def load_image_v2(img_path):
    test_img = cv2.imread(img_path)[:, :, ::-1].astype(np.uint8)
    h, w, c = test_img.shape
    h_mult = int(h // RESIZE)
    w_mult = int(w // RESIZE)
    h_crop = h_mult * RESIZE
    w_crop = w_mult * RESIZE
    dh = (h - h_crop) // 2
    dw = (w - w_crop) // 2
    test_img = test_img[dh : dh + h_crop, dw : dw + w_crop, :]
    print(test_img.shape)
    test_img = cv2.resize(test_img, (RESIZE, RESIZE), interpolation=cv2.INTER_AREA)
    print(test_img.dtype)
    print(test_img.shape)
    delta = (RESIZE - INPUT_SIZE) // 2
    print(delta)
    test_img = test_img[delta : delta + INPUT_SIZE, delta : delta + INPUT_SIZE, :]
    print(test_img.shape)
    cv2.imwrite('test_image_192.png', test_img[:, :, ::-1], [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    raw_img = test_img.copy()
    test_img = test_img.astype(np.float32) / 255.0
    test_img = np.transpose(test_img, (2, 0, 1))[np.newaxis, ...]
    return test_img, raw_img

def load_image_pytorch(img_path):
    test_img = cv2.imread(img_path)[:, :, ::-1].astype(np.uint8).transpose(2, 0, 1)[np.newaxis, ...]
    test_img = torch.FloatTensor(test_img) / 255.0
    test_img = F.interpolate(test_img, size=(RESIZE, RESIZE), mode='area')
    return test_img.data.numpy(), None

def load_image_pre(img_path):
    test_img = cv2.imread(img_path)[:, :, ::-1].astype(np.uint8)
    h, w, c = test_img.shape
    assert h == w and h == INPUT_SIZE
    raw_img = test_img.copy()
    test_img = test_img.astype(np.float32) / 255.0
    test_img = np.transpose(test_img, (2, 0, 1))[np.newaxis, ...]
    return test_img, raw_img

def prediction_transfer(model_outputs):
    """
    Transfer model prediction to normalized scores and rearrange them.
    Args:
        model_outputs: the output of the model, which is a vector with 7
        elements. Note that the saturation is the last element in this vector.
    Returns:
        The final scores of all 7 defects in a vector.
    """
    trans_outputs = model_outputs[:-1]  # take the first 6 defects
    sat_output = model_outputs[-1]      # take the last defect (saturation)

    x_value = [x for x in range(6)]
    y_value = [0., 0.05, 0.15, 0.3, 0.55, 1.0]
    linear_inter = lambda x_0, y_0, x_1, y_1, x: y_0 + (y_1 - y_0) / (x_1 - x_0) * (x - x_0)
    value_trans = []
    for value in trans_outputs:
        if value <= 1:
            value = linear_inter(x_value[0], y_value[0], x_value[1], y_value[1], value)
        elif 1 < value <= 2:
            value = linear_inter(x_value[1], y_value[1], x_value[2], y_value[2], value)
        elif 2 < value <= 3:
            value = linear_inter(x_value[2], y_value[2], x_value[3], y_value[3], value)
        elif 3 < value <= 4:
            value = linear_inter(x_value[3], y_value[3], x_value[4], y_value[4], value)
        else:
            value = linear_inter(x_value[4], y_value[4], x_value[5], y_value[5], value)
        value_trans.append(value)

     # rearrange to match with the predefined order
    return value_trans[:2] + [sat_output] + value_trans[2:]

def softmax_prediction_transfer(other, sat):
    print(other.shape)
    print(sat.shape)

    peak = torch.range(0, 1, 0.1)
    peak_sat = torch.range(-1, 1, 0.1)

    other_pred = F.softmax(other)
    sat_pred = F.softmax(sat)

    other_pred = other_pred * peak
    sat_pred = sat_pred * peak_sat

    return torch.cat((other_pred[:2, :].sum(dim=1), sat_pred.sum(dim=1), other_pred[2:, :].sum(dim=1)), dim=0)

def test_single_caffe(caffe_prototxt, caffe_model, test_img):

    caffe_net = caffe.Net(caffe_prototxt, caffe_model, caffe.TEST)

    forward_kwargs = {
        'data': test_img,
        'stage1_dummy': np.zeros([1, 24, INPUT_SIZE // 4, INPUT_SIZE // 4]),
        'stage2_0_dummy': np.zeros([1, 216, INPUT_SIZE // 8, INPUT_SIZE // 8]),
        'stage3_0_dummy': np.zeros([1, 240, INPUT_SIZE // 16, INPUT_SIZE // 16])
    }
    if not TRUNC_STAGE:
        forward_kwargs['stage4_0_dummy'] = np.zeros(
            [1, 480, INPUT_SIZE // 32, INPUT_SIZE // 32])

    caffe_net.forward(**forward_kwargs)

    # blob_out = caffe_net.forward(**forward_kwargs).copy()
    # for blob in caffe_net.blobs:
    #     if 'split' not in blob:
    #         # print(blob)
    #         caffe_net.blobs[blob].data.tofile('caffe_blobs/%s.bin' % blob)


    if HEAD_TYPE == 'cls':
        if HYBRID:
            # DEBUG
            # print(caffe_net.blobs['hybrid_classifiers_0'].data.shape)
            # print(caffe_net.blobs['hybrid_unsqueeze_0'].data.shape)
            # print(caffe_net.blobs['hybrid_concat'].data.shape)

            hybrid_result = SELECT_HYBRID_BRANCH_CAFFE(caffe_net.blobs)[0]


            # blob_name = 'hybrid_sum'
            # featmap = caffe_net.blobs[blob_name].data
            # # DEBUG
            # print(featmap.shape)
            #
            # hybrid_result = featmap[0] / HYBRID
            print(hybrid_result.shape)

            result = []
            sat_result = []
            n_pre = 0
            for n in range(OUTPUT_NUM):
                if n != 2:
                    # unsqueeze in dim=1 for same output format
                    result.append(hybrid_result[n_pre:n_pre + SOFTMAX_INPUT_NUM[n]])
                    n_pre += SOFTMAX_INPUT_NUM[n]
                else:
                    sat_result.append(hybrid_result[n_pre:n_pre + SOFTMAX_INPUT_NUM[n]])
                    n_pre += SOFTMAX_INPUT_NUM[n]

            other = torch.FloatTensor(np.stack(result, axis=0))
            sat = torch.FloatTensor(np.stack(sat_result))
            pred = softmax_prediction_transfer(other, sat)
            print(pred)
            return pred.data.numpy()
        else:
            # 0.51599044, 0.25606707, -0.07526664, 0.34490782, 0.04331659, 0.10921357, 0.2101593
            blob_name = 'final_concat'
            featmap = caffe_net.blobs[blob_name].data
            pred = prediction_transfer(featmap[0])
            print(pred)
            return pred
    else:
        blob_name = 'final_concat'
        featmap = caffe_net.blobs[blob_name].data
        print(featmap)
        return featmap

def test_single_pytorch(model, ckpt_path, test_img):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # print(test_img.shape)
    test_img = torch.FloatTensor(test_img).to(device)

    model.eval()
    if HYBRID:
        sid = torch.ones(test_img.size(0), dtype=torch.int) * SID
        if HEAD_TYPE == 'reg':
            scores = model(test_img, sid=sid)
            if 2 in SELECTED_DEFECTS:
                idx = SELECTED_DEFECTS.index(2)
                scores = torch.cat([torch.sigmoid(scores[:, :idx]),
                                    scores[:, idx:idx+1],
                                    torch.sigmoid(scores[:, idx+1:])], 1)
            else:
                scores = torch.sigmoid(scores)
        else:
            raw_scores = model(test_img, sid=sid)
            scores = [softmax_prediction_transfer(*raw_scores)]
    else:
        scores = model(test_img)
        if 2 in SELECTED_DEFECTS:
            idx = SELECTED_DEFECTS.index(2)
            scores = torch.cat([torch.sigmoid(scores[:, :idx]),
                                scores[:, idx:idx+1],
                                torch.sigmoid(scores[:, idx+1:])], 1)
        else:
            scores = torch.sigmoid(scores)
    print(scores)
    return scores

class DummyModule(nn.Module):
    def __init__(self):
        super(DummyModule, self).__init__()

    def forward(self, x):
        # print("Dummy, Dummy.")
        return x

def fuse(conv, bn):
    w = conv.weight
    mean = bn.running_mean
    var_sqrt = torch.sqrt(bn.running_var + bn.eps)

    beta = bn.weight
    gamma = bn.bias

    if conv.bias is not None:
        b = conv.bias
    else:
        b = mean.new_zeros(mean.shape)

    w = w * (beta / var_sqrt).reshape([conv.out_channels, 1, 1, 1])
    b = (b - mean)/var_sqrt * beta + gamma

    fused_conv = nn.Conv2d(conv.in_channels,
                         conv.out_channels,
                         conv.kernel_size,
                         conv.stride,
                         conv.padding,
                         groups=conv.groups,
                         bias=True)
    fused_conv.weight = nn.Parameter(w)
    fused_conv.bias = nn.Parameter(b)
    return fused_conv

def fuse_bn_with_conv(m):
    children = list(m.named_children())
    c = None
    cn = None
    # filters = [nn.Sequential, ShuffleUnit, ShuffleNet_Multi_Regression]
    # model_seqs = [module for module in m.modules() if type(module) not in filters]
    for name, child in children:
        if isinstance(child, nn.BatchNorm2d):
            bc = fuse(c, child)
            m._modules[cn] = bc
            m._modules[name] = DummyModule()
            c = None
        elif isinstance(child, nn.Conv2d):
            c = child
            cn = name
        else:
            fuse_bn_with_conv(child)

if __name__ == '__main__':
    # 0.2.5
    HEAD_TYPE = 'reg'
    OUTPUT_NUM = 7
    SELECTED_DEFECTS = [0, 1, 2, 3, 4, 5, 6]
    GROUPS = 8
    GLOBAL_POOLING = 'average'
    TRUNC_STAGE = True
    ARCH_NAME = 'trunc_stage_v2'
    RESIZE = 224
    INPUT_SIZE = 224
    USE_BN = True
    HYBRID = 0
    SID = None
    MODEL_NAME = 'baseline_trunc_stage_v2_balance_setting2'
    CKPT_PATH = '/home/SENSETIME/likaican/Projects/server2/defect-detection-refactor/ckpts' + \
                '/%s/weight_averaged.pkl' % MODEL_NAME

    # 0.2.5 BN-fused
    HEAD_TYPE = 'reg'
    OUTPUT_NUM = 7
    SELECTED_DEFECTS = [0, 1, 2, 3, 4, 5, 6]
    GROUPS = 8
    GLOBAL_POOLING = 'average'
    TRUNC_STAGE = True
    ARCH_NAME = 'trunc_stage_v2_no_bn'
    RESIZE = 224
    INPUT_SIZE = 224
    USE_BN = False
    HYBRID = 0
    SID = None
    MODEL_NAME = 'baseline_trunc_stage_v2_balance_setting2'
    CKPT_PATH = '/home/SENSETIME/likaican/Projects/server2/defect-detection-refactor/ckpts' + \
                '/%s/weight_averaged_fuse.pkl' % MODEL_NAME

    # 0.2.7a
    HEAD_TYPE = 'reg'
    OUTPUT_NUM = 7
    SELECTED_DEFECTS = [0, 1, 2, 3, 4, 5, 6]
    GROUPS = 8
    GLOBAL_POOLING = 'average'
    TRUNC_STAGE = True
    ARCH_NAME = 'trunc_stage_v2_centercrop192'
    RESIZE = 224
    INPUT_SIZE = 192
    USE_BN = True
    HYBRID = 0
    SID = None
    MODEL_NAME = 'baseline_trunc_stage_v2_balance_setting2_centercrop192'
    CKPT_PATH = '/home/SENSETIME/likaican/Projects/server2/defect-detection-refactor/ckpts' + \
                '/%s/weight_averaged.pkl' % MODEL_NAME

    # 0.2.7b
    HEAD_TYPE = 'reg'
    OUTPUT_NUM = 7
    SELECTED_DEFECTS = [0, 1, 2, 3, 4, 5, 6]
    GROUPS = 8
    GLOBAL_POOLING = 'average'
    TRUNC_STAGE = True
    ARCH_NAME = 'trunc_stage_v2_centercrop224_rand_resize'
    RESIZE = 224
    INPUT_SIZE = 224
    USE_BN = True
    HYBRID = 0
    SID = None
    MODEL_NAME = 'baseline_trunc_stage_v2_balance_setting2_centercrop224_rand_resize'
    CKPT_PATH = '/home/SENSETIME/likaican/Projects/server2/defect-detection-refactor/ckpts' + \
                '/%s/weight_averaged.pkl' % MODEL_NAME

    # 0.2.8a
    HEAD_TYPE = 'reg'
    OUTPUT_NUM = 7
    SELECTED_DEFECTS = [0, 1, 2, 3, 4, 5, 6]
    GROUPS = 8
    GLOBAL_POOLING = 'average'
    TRUNC_STAGE = True
    ARCH_NAME = 'trunc_stage_v2_hybrid_v2_no_bn'
    RESIZE = 224
    INPUT_SIZE = 224
    USE_BN = False
    HYBRID = 2
    SID = 0
    MODEL_NAME = 'baseline_trunc_stage_v2_balance_setting2_hybrid2_v2_hdr_pred025_lr1e-3_sat_alpha0p1_decay_interval1_epoch50'
    CKPT_PATH = '/home/SENSETIME/likaican/Projects/server2/defect-detection-refactor/ckpts' + \
                '/%s/weight_averaged.pkl' % MODEL_NAME

    # 0.2.8b
    HEAD_TYPE = 'cls'
    LOGITS_NUM = [11] * 2 + [22] + [11] * 4
    SOFTMAX_INPUT_NUM = [11] * 2 + [21] + [11] * 4
    OUTPUT_NUM = 7
    SELECTED_DEFECTS = [0, 1, 2, 3, 4, 5, 6]
    GROUPS = 8
    GLOBAL_POOLING = 'average'
    TRUNC_STAGE = True
    ARCH_NAME = 'trunc_stage_v2_hybrid_v2_no_bn_cls'
    RESIZE = 224
    INPUT_SIZE = 224
    USE_BN = False
    HYBRID = 3
    SID = 0
    MODEL_NAME = 'baseline_trunc_stage_v2_hybrid3_v2_softmax_classifier'
    CKPT_PATH = '/home/SENSETIME/likaican/Projects/server2/defect-detection-refactor/ckpts' + \
                '/%s/weight_averaged.pkl' % MODEL_NAME
    def select_0_2_8b_caffe(blobs):
        avg_out = np.mean(blobs['hybrid_concat'].data, axis=1)
        return avg_out
    SELECT_HYBRID_BRANCH_TORCH = None
    SELECT_HYBRID_BRANCH_CAFFE = select_0_2_8b_caffe

    # 0.2.8c (the same as 0.2.8b except that the hybrid_1 and hybrid_2 branch of exposure and WB are masked)
    HEAD_TYPE = 'cls'
    LOGITS_NUM = [11] * 2 + [22] + [11] * 4
    SOFTMAX_INPUT_NUM = [11] * 2 + [21] + [11] * 4
    OUTPUT_NUM = 7
    SELECTED_DEFECTS = [0, 1, 2, 3, 4, 5, 6]
    GROUPS = 8
    GLOBAL_POOLING = 'average'
    TRUNC_STAGE = True
    ARCH_NAME = 'trunc_stage_v2_hybrid_v2_no_bn_cls'    # same arch as 0.2.8b
    RESIZE = 224
    INPUT_SIZE = 224
    USE_BN = False
    HYBRID = 3
    SID = 0
    MODEL_NAME = 'v028b_hybrid3_mask'
    CKPT_PATH = '/home/SENSETIME/likaican/Projects/defect-detection/ckpts' + \
                '/%s/weight_averaged.pkl' % MODEL_NAME
    def select_0_2_8c_torch(hybrid_branches_out): # input is of shape (num_hybrids, batch_size, num_softmax_bins)
        avg_out = hybrid_branches_out.mean(dim=0)
        split_point = LOGITS_NUM[0] + LOGITS_NUM[1]
        return torch.cat([hybrid_branches_out[0, :, :split_point], avg_out[:, split_point:]], dim=-1)
    def select_0_2_8c_caffe(blobs):
        hybrid0 = blobs['hybrid_concat'].data[0]
        avg_out = np.mean(blobs['hybrid_concat'].data, axis=1)
        hybrid_bins = np.zeros(shape=hybrid0.shape)
        split_point = LOGITS_NUM[0] + LOGITS_NUM[1]
        hybrid_bins[:, :split_point] = hybrid0[:, :split_point]
        hybrid_bins[:, split_point:] = avg_out[:, split_point:]
        return hybrid_bins
    SELECT_HYBRID_BRANCH_TORCH = select_0_2_8c_torch
    SELECT_HYBRID_BRANCH_CAFFE = select_0_2_8c_caffe

    if HEAD_TYPE == 'reg':
        model = ShuffleNet_Multi_Regression_Pytorch(
            output_num=OUTPUT_NUM, global_pooling_mode=GLOBAL_POOLING,
            trunc_stage=TRUNC_STAGE, groups=GROUPS, hybrid=HYBRID)
    elif HEAD_TYPE == 'cls':
        model = ShuffleNet_Multi_SoftmaxClassifiers_Pytorch(
            output_num=OUTPUT_NUM, global_pooling_mode=GLOBAL_POOLING,
            trunc_stage=TRUNC_STAGE, groups=GROUPS, use_bn=True, hybrid=HYBRID,
            hybrid_branch_selection_func=SELECT_HYBRID_BRANCH_TORCH)
    state_dict = torch.load(CKPT_PATH, map_location='cpu')
    model.load_state_dict(state_dict)

    if not USE_BN:
        if HEAD_TYPE == 'reg':
            model_fuse = ShuffleNet_Multi_Regression_Pytorch(
                output_num=OUTPUT_NUM, global_pooling_mode=GLOBAL_POOLING,
                trunc_stage=TRUNC_STAGE, groups=GROUPS, use_bn=USE_BN, hybrid=HYBRID)
        elif HEAD_TYPE == 'cls':
            model_fuse = ShuffleNet_Multi_SoftmaxClassifiers_Pytorch(
                output_num=OUTPUT_NUM, global_pooling_mode=GLOBAL_POOLING,
                trunc_stage=TRUNC_STAGE, groups=GROUPS, use_bn=USE_BN, hybrid=HYBRID,
                hybrid_branch_selection_func=SELECT_HYBRID_BRANCH_TORCH)

        fuse_bn_with_conv(model)
        model_fuse_dict = model_fuse.state_dict()
        fuse_dict = {k: v for k, v in model.state_dict().items() if k in model_fuse_dict}
        model_fuse_dict.update(fuse_dict)

        # state_dict = model.state_dict()
        # _dict = dict()
        # for key in state_dict:
        #     if '1x1' in key and 'expand' in key:
        #         keyword = key.split('.')
        #         new_key = '.'.join(keyword[:-2] + keyword[-1:])
        #         _dict[new_key] = state_dict[key]
        #     else:
        #         _dict[key] = state_dict[key]
        # state_dict = _dict
        #
        # model_fuse.load_state_dict(state_dict)
        model_fuse.load_state_dict(model_fuse_dict)

        # save fused model
        CKPT_PATH = CKPT_PATH[:-4] + '_fuse.pkl'
        torch.save(model_fuse.state_dict(), CKPT_PATH)

        model = model_fuse

    if not os.path.isdir('prototxts'):
        os.makedirs('prototxts')
    if not os.path.isdir('caffemodels'):
        os.makedirs('caffemodels')

    out_file = 'prototxts/%s.prototxt' % ARCH_NAME
    genAndwrite(out_file, ARCH_NAME)
    print('Successfully generated prototxt: %s' % out_file)

    caffemodel_path = 'caffemodels/%s.caffemodel' % ARCH_NAME
    gen_caffemodel(out_file, CKPT_PATH, caffemodel_path)
    print('Successfully generated caffemodel: %s' % caffemodel_path)

    input('Wait for modifying prototxt, press enter to continue')

    score_file = open('test_scores.csv', 'w')

    score_list = []
    diff_list = []
    for img_file in glob.glob('align_samples/*.jpg'):
        img_dict = preprocess_v2(img_file)
        img_id = img_file.split('/')[-1][:-4]
        cv2.imwrite('preprocessed_data/%s_crop.png' % img_id, img_dict['crop'][:, :, ::-1])
        cv2.imwrite('preprocessed_data/%s_224.png' % img_id, img_dict['after_area'][:, :, ::-1])
        test_img = convert_to_tensor(img_dict['after_area'])
        caffe_result = test_single_caffe('prototxts/%s_caffe_test.prototxt' % ARCH_NAME, caffemodel_path, test_img)
        pytorch_result = test_single_pytorch(model, CKPT_PATH, test_img)[0].data.numpy()
        score_str = list(map(str, list(pytorch_result)))

        score_list.append([img_file.split('/')[-1]] + score_str)
        diff = np.round(np.abs(caffe_result - pytorch_result), decimals=4)
        diff_list.append(diff)
        print('Difference between caffe and pytorch results:')
        print(diff)
    score_list.sort(key=lambda x: x[0])
    for line in score_list:
        score_file.write(','.join(line) + '\n')
    score_file.close()

    print('Summary: mean absolute difference =', np.mean(diff_list, axis=0))
    print('Please check if the above difference is zero, if so, you may proceed to modify the prototxt to match the packaging format')
