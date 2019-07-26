import numpy as np
from scipy import logical_and
from scipy.stats import spearmanr

from auto_test_and_gen_report import auto_test_and_gen_from_model


def get_score_summary(scores, version):
    if version == 1:
        func = summary_v1
    else:
        raise ValueError(f'Score summary version {version} not found.')
    return func(scores)


def summary_v1(scores):
    """ 综合得分计算公式
    result =  0.1120*(1-Math.pow(exposure,1.2))+ \
              0.1179*(1-Math.pow(Math.abs(saturation+0.06),1.2))+ \
              0.0242*(1-Math.pow(noise,1.2))+ 0.0940*(1-Math.pow(haze,1.2))+ 0.3220*(1-blur)+0.1484*(1-balance)
    """
    return 0.1120*(1 - np.power(scores[0], 1.2))+0.1179*(1-np.power(np.abs(scores[2]+0.06), 1.2)) + \
           0.0242*(1-np.power(scores[3], 1.2))+0.0940*(1-np.power(scores[4], 1.2))+0.3220*(1-scores[5])+0.1484*(1-scores[1])


def compute_ranking_accuracy(config, score_dict, mode, summary_version=None):
    score_dict_by_attr = dict()
    for img_path in score_dict:
        attr_name, id, img_file = img_path.split('/')[-3:]

        if attr_name not in score_dict_by_attr:
            score_dict_by_attr[attr_name] = {id: []}
        elif id not in score_dict_by_attr[attr_name]:
            score_dict_by_attr[attr_name].update({id: []})

        scores = score_dict[img_path]
        if mode == 'sbj':
            summary_score = get_score_summary(scores, summary_version)
            data_sample = [img_file, summary_score, scores]
        elif mode == 'obj':
            indices = [config.selected_defects.index(i) for i in [0, 2, 3, 4, 5, 5]]
            data_sample = [img_file, scores[indices]]
        else:
            raise ValueError()

        score_dict_by_attr[attr_name][id].append(data_sample)

    return auto_test_and_gen_from_model(config.model_name, score_dict_by_attr, mode)


def compute_spearman_rank_corr(score_dict, label_dict, num_defects, sat_idx):
    score_list = [[]] * num_defects
    label_list = [[]] * num_defects
    for img_path in score_dict:
        for i in range(num_defects):
            score_list[i].append(score_dict[img_path][i])
            label_list[i].append(label_dict[img_path][i])

    hist_bin = np.arange(-0.05, 1.1, 0.1)
    hist_bin_saturation = np.array([-1.05, -0.85, -0.65, -0.45, -0.25, -0.05, 0.05, 0.25, 0.45, 0.65, 0.85, 1.05])

    rho_list = []
    for d in range(num_defects):
        y = np.array(score_list[d])
        t = np.array(label_list[d])
        bins = hist_bin_saturation if d == sat_idx else hist_bin
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
                print(i, bins[i], bins[i+1])
        idx_samples_array = np.concatenate(idx_samples_list, axis=1)

        samples_list = []
        for i in range(num_samples):
            idx_samples = idx_samples_array[i, :]
            t_samples = t[idx_samples]
            y_samples = y[idx_samples]
            [correlation_samples, p_value] = spearmanr(t_samples, y_samples)
            samples_list.append(correlation_samples)
        rho = np.mean(np.array(samples_list))
        rho_list.append(rho)
    rho_list.insert(0, np.mean(np.array(rho_list)))
    rho_list = np.round(rho_list, decimals=4)
    return list(rho_list)


def evaluate_blur_ranking_accuracy(wrap, test_dataloader, blur_idx):
    score_dict, _, _ = wrap.gather(test_dataloader, gather_loss=False)

    scores_list = sorted(score_dict.items(), key=lambda x: x[0])

    num_triplets = len(scores_list) // 3
    scores_list = [[scores_list[n * 3][1][blur_idx],
                    scores_list[n * 3 + 1][1][blur_idx],
                    scores_list[n * 3 + 2][1][blur_idx]]
                    for n in range(num_triplets)]

    all_correct = top2_correct = best2worst = worst2best = 0

    for scores in scores_list:
        sorted_scores = sorted(scores)
        if sorted_scores == scores:
            all_correct += 1
        if sorted_scores[-2:] == sorted(scores[-2:]):
            top2_correct += 1
        if sorted_scores[0] == scores[-1]:
            best2worst += 1
        if sorted_scores[-1] == scores[0]:
            worst2best += 1

    all_cnt = len(scores_list)                      # rate by random guess
    rate_all_correct = all_correct / all_cnt        # 0.1667
    rate_top2_correct = top2_correct / all_cnt      # 0.3333
    rate_best2worst = best2worst / all_cnt          # 0.3333
    rate_worst2best = worst2best / all_cnt          # 0.3333
    score_summary = rate_all_correct * 0.2 + (1 - rate_best2worst) * 0.3 + \
                    (1 - rate_worst2best) * 0.5     # 0.5667
    print('all: %.2f | top2: %.2f | b2w: %.2f | w2b: %.2f | summary: %.2f' %
          (rate_all_correct, rate_top2_correct, rate_best2worst,
           rate_worst2best, score_summary))

    eval_dict = {
        'rate_all_correct': rate_all_correct,
        'rate_top2_correct': rate_top2_correct,
        'rate_best2worst': rate_best2worst,
        'rate_worst2best': rate_worst2best,
        'score_summary': score_summary
    }

    return eval_dict
