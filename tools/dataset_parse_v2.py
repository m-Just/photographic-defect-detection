# -*- coding:utf-8 -*-
from __future__ import print_function

import os
import csv
import glob
import json
import shutil

import numpy as np
from scipy.stats import spearmanr, pearsonr
from scipy import logical_and
import matplotlib.pyplot as plt


def read_annotations(csv_file):
    annotations = dict()
    with open(csv_file) as f:
        reader = csv.reader(f)
        next(reader) # skip header
        for r in reader:
            img_file = r[0].split('/')[-1]
            annotations[img_file] = list(map(float, r[1:]))
    return annotations

def get_files_recursively(root_path, suffix=['json']):
    assert os.path.isdir(root_path), 'Path "%s" does not exist' % root_path
    files = []
    for path in glob.glob('%s/*' % root_path):
        if os.path.isdir(path):
            files += get_files_recursively(path, suffix)
        elif path.split('.')[-1].lower() in suffix:
            files.append(path)
    return files

def parse_raw_annos(anno_folder, img_root_path, name_dict, val_csv_file=None,
                    check_images=False, remove_outliers=False, strict=True):
    """
    Args:
        anno_folder (str): root directory of the raw annotation files.
        img_root_path (str): root directory of the image files, required only
                             when check_images=True.
        name_dict (dict): define the mapping from the JSON attribute names
                          to the label names.
        val_csv_file (str): path of the csv file containing valid annotation
                            used for validation.
        check_images (bool): whether to check the existence of images.
        remove_outliers (bool): whether to remove outliers of the annotation.
        strict (bool): whether to raise exception during unexpected situations.
    Returns:
        anno_dict (dict): contains the data of new annotations.
        val_dict (dict): contains the data of annotations used for validation.
    """

    anno_dict = {'all': dict()}
    val_img_cnt = 0

    if val_csv_file is not None:
        val_dict = {'all': dict()}
        val_anno = read_annotations(val_csv_file)
    else:
        val_dict = None

    for anno_file in get_files_recursively(anno_folder):
        if anno_file.split('/')[-1] == 'summary.json':
            print('skip summary file %s' % anno_file)
            continue

        with open(anno_file) as f:
            data = json.load(f)
        img_file = data['rawFilename']
        path = data['rawFilePath']

        img_rel_path = ('%s/%s' % (path, img_file)) if path else img_file
        img_full_path = '%s/%s' % (img_root_path, img_rel_path)
        if check_images and not os.path.isfile(img_full_path):
            print('image not found: %s' % img_full_path)
            if strict:
                raise ValueError()

        if val_csv_file is not None and img_file in val_anno:
            val_img_cnt += 1
            print('val image #%d found: %s' % (val_img_cnt, img_file))
            d = val_dict
        else:
            d = anno_dict

        package_name = '/'.join(anno_file[len(anno_folder):].split('/')[1:-2])
        if package_name not in d:
            d[package_name] = dict()
        p = d[package_name]
        d = d['all']

        if img_rel_path not in d:
            d[img_rel_path] = dict()
        if img_rel_path not in p:
            p[img_rel_path] = dict()
        d = d[img_rel_path]
        p = p[img_rel_path]

        for key, name in name_dict.items():
            if key not in data:
                if strict:
                    err_msg = u'Key \"%s\" not found in annotation file \"%s\"'\
                              % (key, anno_file)
                    raise ValueError(err_msg)
                else:
                    continue
            anno = float(data[key])
            if name not in d:
                d[name] = list()
            if name not in p:
                p[name] = list()
            d[name].append(anno)
            p[name].append(anno)

    if remove_outliers:
        for d_ in [anno_dict['all'], val_dict['all']]:
            outlier_cnt = 0
            for img_rel_path in d_:
                d = d_[img_rel_path]
                for name in d:
                    l = d[name]
                    m = sum(l) / len(l)
                    d[name] = [v for v in l if np.abs(v - m) <= 0.5]
                    if len(d[name]) == 0:
                        print('major disagreement from %s (%s):' \
                              % (img_rel_path, name), l)
                        d[name] = l
                    outlier_cnt += len(l) - len(d[name])
            print('%d outlier(s) removed' % outlier_cnt)
    else:
        print('ignore outliers')

    return anno_dict, val_dict

def write_to_csv(dest, label_dict):
    cnt = 0
    score_summary = {k: [] for k in order_list}
    stds = {k: [] for k in order_list}
    with open(dest, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Filename'] + order_list)
        for img_file in label_dict:
            row = [img_file]
            l = label_dict[img_file]
            for d in order_list:
                if len(l[d]) == 0:
                    cnt += 1
                    print('annotation missing #%d:' % cnt)
                    print(img_file)
                    print(l)
                else:
                    score = sum(l[d]) / len(l[d])
                    row.append(score)
                    stddev = np.std(l[d])
                    row.append(stddev)
                    stds[d].append(stddev)
                    score_summary[d].append(score)
            writer.writerow(row)
    return score_summary, stds

def compute_spearmans_rho(score_dict, selected_defects):
    '''
    Args:
        score_dict: key = defect index
                    val = a tuple of the predicted score list and its
                          corresponding ground truth list
        selected_defects: a list of indices representing the number of
                          predicted defects
    '''
    hist_bin = np.arange(-0.05, 1.1, 0.1)
    hist_bin_saturation = np.array([-1.05, -0.85, -0.65, -0.45, -0.25, -0.05, 0.05, 0.25, 0.45, 0.65, 0.85, 1.05])

    rho_list = []
    for i, defect_idx in enumerate(selected_defects):
        y = np.array(score_dict[defect_idx][0])
        t = np.array(score_dict[defect_idx][1])
        if defect_idx == 1:
            bins = hist_bin_saturation
        else:
            bins = hist_bin
        hist = np.histogram(t, bins=bins)[0]

        num_samples = len(t) * 10
        idx_samples_list = []
        for i in range(len(hist)):
            idx = np.where(logical_and(t >= bins[i], t < bins[i+1]))[0]
            if len(idx) > 0:
                idx_samples_array = np.zeros((num_samples, 1), dtype=np.uint32)
                idx_samples = np.random.choice(idx, num_samples)
                idx_samples_array[:, 0] = idx_samples
                idx_samples_list.append(idx_samples_array)
            else:
                print('no annotation found in the interval', defect_idx, bins[i], bins[i+1])
        idx_samples_array = np.concatenate(idx_samples_list, axis=1)

        samples_list = []
        for i in range(num_samples):
            idx_samples = idx_samples_array[i, :]
            t_samples = t[idx_samples]
            y_samples = y[idx_samples]
            [correlation_samples, p_value] = spearmanr(t_samples, y_samples)
            # if np.isnan(correlation_samples):
            #     print(y_samples)
            #     print(t_samples)
            #     raw_input('press enter to continue')
            samples_list.append(correlation_samples)
        rho = np.mean(np.array(samples_list))
        rho_list.append(rho)
    rho_list.insert(0, np.mean(np.array(rho_list)))
    return rho_list

def compute_spearmans_rho_v2(score_dict, selected_defects):
    rho_list = []
    for i, defect_idx in enumerate(selected_defects):
        y = np.array(score_dict[defect_idx][0])
        t = np.array(score_dict[defect_idx][1])
        correlation, p_value = spearmanr(t, y)
        rho_list.append(correlation)
    rho_list.insert(0, np.mean(np.array(rho_list)))
    return rho_list

def compute_pearson_corr(score_dict, selected_defects):
    rho_list = []
    for i, defect_idx in enumerate(selected_defects):
        y = np.array(score_dict[defect_idx][0])
        t = np.array(score_dict[defect_idx][1])
        correlation, p_value = pearsonr(t, y)
        rho_list.append(correlation)
    rho_list.insert(0, np.mean(np.array(rho_list)))
    return rho_list

def compute_l2_dist(score_dict, selected_defects):
    rho_list = []
    for i, defect_idx in enumerate(selected_defects):
        y = np.array(score_dict[defect_idx][0])
        t = np.array(score_dict[defect_idx][1])
        dist = np.sqrt(np.mean((y - t) ** 2))
        rho_list.append(dist)
    rho_list.insert(0, np.mean(np.array(rho_list)))
    return rho_list

def compare_spearmans_rho(new_csv_file, ref_csv_file, img_root_path=None, copy_samples=True):
    new_anno = read_annotations(new_csv_file)
    ref_anno = read_annotations(ref_csv_file)
    assert set(new_anno.keys()).issubset(set(ref_anno.keys()))

    file_list = sorted(new_anno.keys())
    score_dict = {k: ([], []) for k in range(len(order_list))}
    for i, defect in enumerate(order_list):
        for j, img_file in enumerate(file_list):
            score_dict[i][0].append(new_anno[img_file][i])
            score_dict[i][1].append(ref_anno[img_file][all_defects.index(defect)])
            if i == 0 and j < 50:
                if copy_samples:
                    shutil.copyfile('%s/%s' % (img_root_path, img_file), 'samples/%02d.jpg' % j)
        assert len(score_dict[i][0]) > 0
        assert len(score_dict[i][1]) > 0

    indices = range(len(order_list))
    corr_list_v1 = compute_spearmans_rho(score_dict, indices)
    corr_list_v2 = compute_spearmans_rho_v2(score_dict, indices)
    corr_list_v3 = compute_pearson_corr(score_dict, indices)
    corr_list_v4 = compute_l2_dist(score_dict, indices)

    return corr_list_v1, corr_list_v2, corr_list_v3, corr_list_v4

if __name__ == '__main__':
    # img_root_path = 'dataset/test'
    # label_folder = 'data/raw/flickr_defect_test_labels'
    # name_dict = {u'亮度缺陷': 'Bad Exposure',
    #              u'噪声': 'Noise',
    #              u'模糊': 'Undesired Blur',
    #              u'雾感': 'Haze',
    #              u'饱和度': 'Bad Saturation'}
    #
    # csv_dest = 'labels.csv'
    # order_list = ['Bad Exposure', 'Bad Saturation', 'Noise', 'Haze', 'Undesired Blur']
    #
    # all_defects = ['Bad Exposure', 'Bad White Balance', 'Bad Saturation', 'Noise',
    #                'Haze', 'Undesired Blur', 'Bad Composition']
    #
    # label_dict = load_labels(label_folder, img_root_path)
    # print(label_dict.items()[0])
    #
    # write_to_csv(csv_dest, label_dict)
    #
    # compare_spearmans_rho('labels.csv', 'data/test/defect_testing_gt_new.csv')

    # img_root_path = 'dataset'
    # label_folder = 'data/raw/dof_blur'
    # name_dict = {'DOFBlur': 'DOF Blur'}
    #
    # csv_dest = 'dof_blur_labels.csv'
    # order_list = ['DOF Blur']
    #
    # all_defects = ['DOF Blur']
    #
    # label_dict = load_labels(label_folder, img_root_path)
    # print(label_dict.items()[0])
    #
    # write_to_csv(csv_dest, label_dict)

    # compare_spearmans_rho('labels.csv', 'data/test/defect_testing_gt_new.csv')

    # img_root_path = 'dataset/test'
    # label_folder = 'data/raw/flickr_defect_test_labels_ours'
    # name_dict = {u'亮度缺陷': 'Bad Exposure',
    #              u'噪声': 'Noise',
    #              u'模糊': 'Undesired Blur',
    #              u'雾感': 'Haze',
    #              u'饱和度': 'Bad Saturation',
    #
    #              u'?和度': 'Bad Saturation',
    #              u'?感': 'Haze',
    #              u'噪?': 'Noise',
    #              }
    #
    # csv_dest = 'flickr_labels_ours.csv'
    # order_list = ['Bad Exposure', 'Bad Saturation', 'Noise', 'Haze', 'Undesired Blur']
    #
    # all_defects = ['Bad Exposure', 'Bad White Balance', 'Bad Saturation', 'Noise',
    #                'Haze', 'Undesired Blur', 'Bad Composition']
    # label_dict = load_labels(label_folder, img_root_path)
    # print('example:', label_dict.items()[0])
    #
    # write_to_csv(csv_dest, label_dict)
    #
    # compare_spearmans_rho(csv_dest, 'data/test/defect_testing_gt_new.csv')

    img_root_path = 'dataset/test'
    val_csv_file = 'data/test/defect_testing_gt_new.csv'
    # val_csv_file = 'data/flickr_labels_ours.csv'
    val_img_root_path = 'dataset/test'
    label_folder = 'data/raw/selected_0603'
    # label_folder = 'data/raw/flickr_defect_test_labels_ours'

    name_dict = {u'亮度缺陷': 'Bad Exposure',
                 u'噪声': 'Noise',
                 u'模糊': 'Undesired Blur',
                 u'雾感': 'Haze',
                 u'饱和度': 'Bad Saturation',
                 u'?和度': 'Bad Saturation',
                 u'?感': 'Haze',
                 u'噪?': 'Noise',}

    csv_dest = 'hdr_labels_0626.csv'
    order_list = ['Bad Exposure', 'Bad Saturation', 'Noise', 'Haze', 'Undesired Blur']
    all_defects = ['Bad Exposure', 'Bad White Balance', 'Bad Saturation', 'Noise',
                   'Haze', 'Undesired Blur', 'Bad Composition']
    # all_defects = ['Bad Exposure', 'Bad Saturation', 'Noise', 'Haze', 'Undesired Blur']

    # img_root_path = None
    # label_folder = 'data/raw/selected_0603_v2'
    #
    # name_dict = {u'白平衡': 'Bad White Balance'}
    # order_list = ['Bad White Balance']
    #
    # val_csv_file = 'data/test/defect_testing_gt_new.csv'
    # val_img_root_path = 'dataset/test'
    # all_defects = ['Bad Exposure', 'Bad White Balance', 'Bad Saturation', 'Noise',
    #                'Haze', 'Undesired Blur', 'Bad Composition']
    #
    # csv_dest = 'hdr_labels_wb.csv'

    # start parsing
    anno_dict, val_dict = parse_raw_annos(
        label_folder, img_root_path, name_dict, val_csv_file=val_csv_file,
        check_images=False, remove_outliers=True, strict=False)

    print('summary:')
    for k, l in sorted(anno_dict.items(), key=lambda x: x[0]):
        print('new(%s): %d' % (k, len(l)))
    for k, l in sorted(val_dict.items(), key=lambda x: x[0]):
        print('val(%s): %d' % (k, len(l)))
    if len(anno_dict['all']) > 0:
        print('new example:', anno_dict['all'].items()[0])
    if len(val_dict['all']) > 0:
        print('val example:', val_dict['all'].items()[0])

    # save new annotation parsed results
    score_summary, stds = write_to_csv(csv_dest, anno_dict['all'])
    for name in order_list:
        plt.hist(score_summary[name], bins=50)
        plt.savefig('label_packages/%s.png' % name)
        plt.clf()
        plt.hist(stds[name], bins=50)
        plt.savefig('label_packages/std_%s.png' % name)
        plt.clf()

    # validate with the reference score
    if val_dict is not None:
        root_dir = 'label_packages'
        if not os.path.isdir(root_dir):
            os.makedirs(root_dir)

        packages = []
        for package_name in val_dict:
            val_csv_dest = '%s/%s.csv' % (root_dir, '_'.join(package_name.split('/')))
            write_to_csv(val_csv_dest, val_dict[package_name])

            corr_list = compare_spearmans_rho(val_csv_dest, val_csv_file, val_img_root_path, copy_samples=False)

            packages.append([])
            packages[-1].append('%s: %d images\n' % (package_name, len(val_dict[package_name])))
            for i, defect in enumerate(order_list):
                corr = tuple([c[i+1] for c in corr_list])
                packages[-1].append('%-30s' % defect + '%-15.4f%-15.4f%-15.4f%.4f\n' % corr)
            packages[-1].append('\n')
        packages.sort(key=lambda x: x[0])

        with open('%s/summary.txt' % root_dir, 'w') as f:
            for lines in packages:
                for line in lines:
                    f.write(line)
