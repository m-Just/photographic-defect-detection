# coding=utf-8
import os
import io
from collections import OrderedDict

from utils import get_defect_names_by_idx


# sort a dictionary by value and return the keys
def sort_by_value(d, reverse):
    items=d.items()
    backitems=[[v[1],v[0]] for v in items]
    # 将得分按照从大到小排序
    backitems.sort(reverse=reverse)
    return [ backitems[i][1] for i in range(0,len(backitems))]


# report head
def gen_report_head(handle, title):
    handle.write(u'<!DOCTYPE html>\n')
    handle.write(u'<html>\n')
    handle.write(u'<head><title>' + title + '</title></head>\n')
    handle.write(u'<body>\n')
    handle.write(u'<h1>' + title + '</h1>')
    handle.write(title)


# report tail
def gen_report_tail(handle):
    handle.write(u'</body>\n')
    handle.write(u'</html>\n')


# report (core)
def gen_report_table(handle, scores, info, reverse):
    defect_names = get_defect_names_by_idx()

    file_path = info['file_path']
    group_size = info['group_size']

    all_correct_cnt = 0
    top2_correct_cnt = 0
    best2worst_cnt = 0
    worst2best_cnt = 0
    all_cnt = 0

    handle.write(u'<table style="width:100%" border="1">')

    # names of all groups in a single evaluation
    names = scores.keys()
    for name in names:
        # sheet = scores.sheet_by_name(name)
        sheet = scores[name]
        # print(len(sheet))
        # for sheet_item in sheet:
        #     print(len(sheet_item))
        #     print(sheet_item[0])
        imgs_path = os.path.join(file_path, name)
        print('Check: ' + imgs_path)
        imgs = os.listdir(imgs_path)

        # sort by names, then this is the GT order
        imgs.sort()
        # find the scores for each image (one attribute)
        img_score = {}
        for i in range(group_size):
            img_score[sheet[i][0]] = abs(float(sheet[i][1]))
        img_score_sorted = sort_by_value(img_score, reverse)

        # check if the order is the same with GT
        all_correct = 1
        for i in range(group_size):
            # print(imgs[i] + ' vs ' + img_score_sorted[i] )
            if imgs[i] != img_score_sorted[i]:
                all_correct = 0
                break
        all_correct_cnt += all_correct
        all_cnt += 1
        # print(img_score)

        if all_correct or \
           (imgs[0] == img_score_sorted[0] and imgs[1] == img_score_sorted[1]) or \
           (imgs[0] == img_score_sorted[1] and imgs[1] == img_score_sorted[0]):
           top2_correct_cnt +=  1

        if not all_correct and imgs[0] == img_score_sorted[-1]:
            best2worst_cnt += 1

        if not all_correct and imgs[-1] == img_score_sorted[0]:
            worst2best_cnt += 1

        # show this group of images if there is error
        if all_correct == 0:
            # put images
            handle.write(u'<tr>\n')
            for img in imgs:
                # handle.write('<th><div id="container"><img src="' + os.path.join(imgs_path, img) + '"/></div></th>\n')
                handle.write(u'<th><img src="' + os.path.join(imgs_path, img) + '" width=320px></th>\n')
            handle.write(u'</tr>\n')

            # put scores
            handle.write(u'<tr>\n')
            for img in imgs:
                # find its score
                for i in range(group_size):
                    if img == sheet[i][0]:
                        break
                handle.write(u'<th>\n')
                handle.write(img + ':' + str(sheet[i][1]))
                if len(sheet[i]) > 2:
                    for j in range(len(sheet[i][2])):
                        handle.write(u'<br>\n' + defect_names[j] + ':'
                                               + str(sheet[i][2][j]))
                handle.write('\n</th>\n')
            handle.write(u'</tr>\n')

    handle.write(u'</table>')

    rate_all_correct = 1.0 * all_correct_cnt / all_cnt
    rate_top2_correct = 1.0 * top2_correct_cnt / all_cnt
    rate_best2worst = 1.0 * best2worst_cnt / all_cnt
    rate_worst2best = 1.0 * worst2best_cnt / all_cnt
    score_summary = rate_all_correct * 0.2 + (1-rate_best2worst) * 0.3 + (1-rate_worst2best) * 0.5
    summary  = f'排序命中率： {rate_all_correct}\n'       # 排序全符合标定的组数/总组数*100%
    summary += f'排序命中率(Top2)： {rate_top2_correct}\n'# Top2命中人工Top2的概率
    summary += f'排序错分率： {rate_best2worst}\n'        # 排序最高分为标定最低分的组数/总组数*100%
    summary += f'综合错分率： {rate_worst2best}\n'        # 标定低质量照片被认为是高质量照片的图片数/总图片数*100%
    summary += f'综合分： {score_summary}\n'             # 综合分=排序命中率*20%+（1-排序错分率）*30%+（1-综合错分率）*50%

    print(summary)

    return summary, [rate_all_correct, rate_top2_correct, rate_best2worst,
                     rate_worst2best, score_summary]


# 测试单个属性，并生成结果摘要
def test_single_attribute(data_list, testset_dir, num_groups_list,
                          group_size_list, score_name_list, attribute_id,
                          items, save_path, sbj_test):

    # attribute_id = 0
    score_file = os.path.join(save_path, score_name_list[attribute_id])
    if not os.path.isdir(score_file):
        os.makedirs(score_file)
    file_path = os.path.join(testset_dir, data_list[attribute_id])

    group_size = group_size_list[attribute_id]
    num_groups = num_groups_list[attribute_id]
    report_file = score_file + '/' + score_name_list[attribute_id] + '_check_result_report.html'

    info = {}
    # without use score file
    # info['score_file'] = score_file
    info['file_path'] = file_path
    info['group_size'] = group_size
    info['num_groups'] = num_groups

    # scores = load_workbook(score_file)
    # scores = open_workbook(score_file)
    scores = items
    print(data_list[attribute_id])
    print(score_file)
    print(report_file)
    report = io.open(report_file, 'w+', encoding='utf-8')
    title = data_list[attribute_id]
    gen_report_head(report, title)
    summary, rate = gen_report_table(report, scores, info, sbj_test)
    gen_report_tail(report)

    report.close()

    summary_head = f'\nAttribute: {data_list[attribute_id]}\n'
    summary_head = summary_head + f'Score file: {score_file}\n'
    summary_head = summary_head + f'Image path: {file_path}\n'
    summary = summary + f'Failure images: {report_file}\n'

    return summary_head + summary + '\n', rate


# compute scores and show result from model outputs
def auto_test_and_gen_from_model(model_name, dicts, mode):
    report_path = f'reports/{model_name}'
    if not os.path.isdir(report_path):
        os.makedirs(report_path)

    if mode == 'sbj':
        data_list = ['风景', '人物', '活动', '宠物', '植物', '美食']
        testset_dir = './dataset/subjective_testing_public'  # 测试集目录
        num_groups_list = [20] * len(data_list)  # 每个子数据集有多少组图片
        group_size_list = [3] * len(data_list)  # 每组图片几张图, 主观测试数据集
        score_name_list = data_list  # 打分表名字
        test_prefix = 'sbj_test'
        sbj_test = True

    elif mode == 'obj':
        data_list = ['曝光补偿', '饱和度', '噪点', '雾感', '模糊', '人像模糊']
        testset_dir = './dataset/objective_testing_mod'  # 测试集目录
        num_groups_list = [35, 30, 30, 33, 41, 27]  # 每个子属性有多少组图片
        group_size_list = [3, 3, 3, 3, 3, 3]  # 每组图片几张图, 新的客观数据集
        score_name_list = ['baoguang', 'baohedu', 'zaodian', 'wugan', 'mohu', 'renxiang_mohu']  # 打分表名字
        test_prefix = 'obj_test'
        sbj_test = False
        for i, key in enumerate(data_list):
            for id in dicts[key]:
                dicts[key][id] = [[img_file, scores[i]] for img_file, scores in dicts[key][id]]

    elif mode == 'sbj_deprecated':
        data_list = ['oppo图集', 'ST图集']
        testset_dir = './dataset/subjective_testing'  # 测试集目录
        num_groups_list = [22, 80]  # 每个子数据集有多少组图片
        group_size_list = [3, 3]  # 每组图片几张图, 主观测试数据集
        score_name_list = ['oppo图集', 'ST图集']  # 打分表名字
        test_prefix = 'sbj_deprecated_test'
        sbj_test = True

    elif mode == 'obj_deprecated':
        data_list = ['曝光补偿', '饱和度', '噪点', '雾感', '模糊', '人像模糊']
        testset_dir = './dataset/objective_testing'  # 测试集目录
        num_groups_list = [30, 30, 30, 30, 30, 5]  # 每个子属性有多少组图片
        group_size_list = [3, 2, 3, 3, 3, 3]  # 每组图片几张图, 新的客观数据集
        score_name_list = ['baoguang', 'baohedu', 'zaodian', 'wugan', 'mohu', 'renxiang_mohu']  # 打分表名字
        test_prefix = 'obj_deprecated_test'
        sbj_test = False
        for i, key in enumerate(data_list):
            for id in dicts[key]:
                dicts[key][id] = [[img_file, scores[i]] for img_file, scores in dicts[key][id]]

    summary_txt_path = f'{report_path}/{test_prefix}_report.txt'
    summary_txt = io.open(summary_txt_path, 'w+', encoding='utf-8')
    summary_txt.write(u'测试集目录：' + testset_dir + '\n')

    rates = OrderedDict()
    for i, key in enumerate(data_list):
        samples_path = f'{report_path}/{test_prefix}_samples'
        summary, rate = test_single_attribute(
            data_list, testset_dir, num_groups_list, group_size_list, score_name_list,
            i, dicts[key], samples_path, sbj_test)
        summary_txt.writelines(summary)
        rates[key] = rate

    summary_txt.close()
    print('Summary written to: ' + summary_txt_path)
    return rates
