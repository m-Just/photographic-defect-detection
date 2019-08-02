#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import glob
import json
import numpy as np
import matplotlib.pyplot as plt

import pprint
pp = pprint.PrettyPrinter()

data_folder = '/home/SENSETIME/likaican/defect_data/数据/defects'
label_folder = '%s/defect_label2' % data_folder
image_folder = '%s/defect_data/defect_detection_separate' % data_folder

sample_num = 100 # for each defect annotated by each annotator
defect_list = ['Bad Exposure', 'Bad White Balance', 'Bad Saturation', 'Noise', 'Haze', 'Undesired Blur', 'Bad Composition']

# %%
fig = plt.figure(figsize=(20,30))

defects1 = dict([(defect_name, []) for defect_name in defect_list])
with open('./data/train/defect_training_gt_new.csv') as f:
    f.readline()
    for line in f.readlines():
        data = map(float, line.strip()[1:-1].split('","')[1:])
        for i in range(len(defect_list)):
            defects1[defect_list[i]].append(data[i])
for i, defect_name in enumerate(defect_list):
    print(defect_name)
    ax = plt.subplot(7, 3, i*3+1)
    ax.set_ylabel(defect_name)
    ax.hist(defects1[defect_name], bins=100)

defects2 = dict([(defect_name, []) for defect_name in defect_list])
with open('./data/all.csv') as f:
    f.readline()
    for line in f.readlines():
        data = map(float, line.strip()[1:-1].split('","')[1:])
        for i in range(len(defect_list)):
            defects2[defect_list[i]].append(data[i])
for i, defect_name in enumerate(defect_list):
    print(defect_name)
    ax = plt.subplot(7, 3, i*3+2)
    ax.hist(defects2[defect_name], bins=100)

for i, defect_name in enumerate(defect_list):
    print(defect_name)
    ax = plt.subplot(7, 3, i*3+3)
    ax.hist(defects1[defect_name] + defects2[defect_name], bins=100)
plt.savefig('plot.png')


# defects = dict([(defect_name, []) for defect_name in defect_list])
# # with open('./data/train/defect_training_gt_new.csv') as f:
# with open('./data/new.csv') as f:
#     f.readline()
#     for line in f.readlines():
#         data = map(float, line.strip()[1:-1].split('","')[1:])
#         for i in range(len(defect_list)):
#             defects[defect_list[i]].append(data[i])
#         # print(defects)
# for defect_name in defect_list:
#     print(defect_name)
#     ax = fig.add_subplot(111)
#     ax.hist(defects[defect_name], bins=100)

# %%
defects = dict([(defect_name, []) for defect_name in defect_list])
for subdir in glob.glob('%s/*separate*' % label_folder):
    if 'sampled' in subdir: continue
    for i, label_file in enumerate(glob.glob('%s/Label/*.json' % subdir)):
        if i >= sample_num: break
        with open(label_file) as f:
            data = json.load(f)
        try:
            attr_type = data['image']['rawFilePath']
            attr_val = data['objects']['imageAsObj'][0]['attributes'].values()[0]['value']
            defects[attr_type].append(attr_val)
        except:
            pass
for defect_name in defects:
    print(len(defects[defect_name]))

# %%
label_by_person = dict([(i, dict()) for i in range(6)])
for subdir in glob.glob('%s/*separate*' % label_folder):
    if 'sampled' in subdir: continue
    person_id = int(subdir[-1]) if subdir[-1] != 'o' else 0
    attr_type = subdir.split('_')[-3] if subdir[-1] != 'o' else subdir.split('_')[-2]
    print(attr_type)
    for i, label_file in enumerate(glob.glob('%s/Label/*.json' % subdir)):
        if i >= sample_num: break
        with open(label_file) as f:
            data = json.load(f)
            # pp.pprint(data)
        try:
            # attr_type = data['objects']['imageAsObj'][0]['attributes'].keys()[0]
            # attr_type = data['image']['rawFilePath']
            data_dict = {
                'img_id': data['image']['id'],
                'img_filename': data['image']['rawFilename'],
                'attr_val': data['objects']['imageAsObj'][0]['attributes'][data['objects']['imageAsObj'][0]['attributes'].keys()[0]]['value']
            }
            if attr_type not in label_by_person[person_id]:
                label_by_person[person_id][attr_type] = []
            label_by_person[person_id][attr_type].append(data_dict)
        except:
            pp.pprint(data)
        # pp.pprint(label_by_person)
print(label_by_person)

# %%
fig = plt.figure(figsize=(12,5))
ax = fig.add_subplot(111)
ind = np.arange(6)
width = 0.2
colors = ['b', 'r', 'g', 'y', 'purple', 'cyan']
names = label_by_person[0].keys()
print(names)
for i in range(6):
    print('Person %d' % i)
    attr_id = -1
    for attr_type in names:
        attr_id += 1
        if attr_type not in label_by_person[i]:
            continue
        attr_list = label_by_person[i][attr_type]
        val_list = []
        for attr in attr_list:
            # pp.pprint(attr)
            val_list.append(float(attr['attr_val']))
        # print(val_list)
        # plt.hist(val_list)
        var = np.var(np.array(val_list))
        print(attr_type, end=' ')
        print('%.4f' % var)

        ax.bar(attr_id + width * (i - 2.5) / 2, var, width/3, color=colors[i], align='center')
ax.axes.set_xticklabels(['']+names)
ax.set_ylabel('Score Variance')
plt.savefig('plot.png')

# %%
names = label_by_person[0].keys()
global_mean = dict()
global_count = dict()
for i in range(6):
    for attr_type in names:
        if attr_type not in label_by_person[i]:
            continue
        attr_list = label_by_person[i][attr_type]
        if attr_type not in global_mean:
            global_mean[attr_type] = sum(map(lambda item: float(item['attr_val']), attr_list))
            global_count[attr_type] = len(attr_list)
        else:
            global_mean[attr_type] += sum(map(lambda item: float(item['attr_val']), attr_list))
            global_count[attr_type] += len(attr_list)
for key in global_mean:
    try:
        global_mean[key] /= float(global_count[key])
    except:
        print(key)
print(global_mean)

# %%
fig = plt.figure(figsize=(12,5))
ax = fig.add_subplot(111)
ind = np.arange(6)
width = 0.2
colors = ['b', 'r', 'g', 'y', 'purple', 'cyan']
names = label_by_person[0].keys()
print(names)
for i in range(6):
    print('Person %d' % i)
    attr_id = -1
    for attr_type in names:
        attr_id += 1
        if attr_type not in label_by_person[i]:
            continue
        attr_list = label_by_person[i][attr_type]
        avg = sum(map(lambda item: float(item['attr_val']) / len(attr_list), attr_list))

        ax.bar(attr_id + width * (i - 2.5) / 2, avg - global_mean[attr_type], width/3, color=colors[i], align='center')
ax.axes.set_xticklabels(['']+names)
ax.set_ylabel('Average score from mean')
plt.savefig('plot.png')
