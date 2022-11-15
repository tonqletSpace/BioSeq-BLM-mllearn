import csv
import math
import os

import numpy as np
import sklearn.metrics as metrics
from scipy.sparse import issparse, lil_matrix
from sklearn.metrics import roc_auc_score


def performance(origin_labels, predict_labels, deci_value, bi_or_multi=False, res=False):
    """evaluations used to evaluate the performance of the model.
    :param deci_value: decision values used for ROC and AUC.
    :param bi_or_multi: binary or multiple classification
    :param origin_labels: true values of the data set.
    :param predict_labels: predicted values of the data set.
    :param res: residue or not.
    """
    if len(origin_labels) != len(predict_labels):
        raise ValueError("The number of the original labels must equal to that of the predicted labels.")
    if bi_or_multi is False:
        tp = 0.0
        tn = 0.0
        fp = 0.0
        fn = 0.0
        for i in range(len(origin_labels)):
            if res is True:
                if origin_labels[i] == 1 and predict_labels[i] == 1:
                    tp += 1.0
                elif origin_labels[i] == 1 and predict_labels[i] == 0:
                    fn += 1.0
                elif origin_labels[i] == 0 and predict_labels[i] == 1:
                    fp += 1.0
                elif origin_labels[i] == 0 and predict_labels[i] == 0:
                    tn += 1.0
            else:
                if origin_labels[i] == 1 and predict_labels[i] == 1:
                    tp += 1.0
                elif origin_labels[i] == 1 and predict_labels[i] == -1:
                    fn += 1.0
                elif origin_labels[i] == -1 and predict_labels[i] == 1:
                    fp += 1.0
                elif origin_labels[i] == -1 and predict_labels[i] == -1:
                    tn += 1.0
        try:
            sn = tp / (tp + fn)
            r = sn
        except ZeroDivisionError:
            sn, r = 0.0, 0.0
        try:
            sp = tn / (fp + tn)
        except ZeroDivisionError:
            sp = 0.0
        try:
            acc = (tp + tn) / (tp + tn + fp + fn)
        except ZeroDivisionError:
            acc = 0.0
        try:
            mcc = (tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        except ZeroDivisionError:
            mcc = 0.0
        try:
            auc = roc_auc_score(origin_labels, deci_value)
        except ValueError:  # modify in 2020/9/13
            auc = 0.0
        try:
            p = tp / (tp + fp)
        except ZeroDivisionError:
            p = 0.0
        try:
            f1 = 2 * p * r / (p + r)
        except ZeroDivisionError:
            f1 = 0.0
        balance_acc = (sn + sp) / 2
        return acc, mcc, auc, balance_acc, sn, sp, p, r, f1
    else:
        correct_labels = 0.0
        for elem in zip(origin_labels, predict_labels):
            if elem[0] == elem[1]:
                correct_labels += 1.0
        acc = correct_labels / len(origin_labels)
        return acc


def mll_performance(origin_labels, predicted_labels):
    """evaluations used to evaluate the performance of the model.
    :param origin_labels: true values of the data set.
    :param predicted_labels: predicted values of the data set.
    """
    if not issparse(origin_labels):
        origin_labels = lil_matrix(origin_labels)
    if not issparse(predicted_labels):
        predicted_labels = lil_matrix(predicted_labels)

    if origin_labels.get_shape() != predicted_labels.get_shape():
        raise ValueError("The shape of the original labels must equal to that of the predicted labels.")

    # 'Ham', 'Acc', 'Jac', 'Pr', 'Rc', 'F1'
    hamming_loss = metrics.hamming_loss(origin_labels, predicted_labels)
    accuracy = metrics.accuracy_score(origin_labels, predicted_labels)
    jaccard_similarity = metrics.jaccard_score(origin_labels, predicted_labels, average='samples', zero_division=0)
    precision = metrics.precision_score(origin_labels, predicted_labels, average='samples', zero_division=0)
    recall = metrics.recall_score(origin_labels, predicted_labels, average='samples', zero_division=0)
    f1_score = metrics.f1_score(origin_labels, predicted_labels, average='micro', zero_division=0)

    return hamming_loss, accuracy, jaccard_similarity, precision, recall, f1_score


def print_metric_dict(results, ind):
    metric_dict = {'Accuracy': results[0], 'MCC': results[1], 'AUC': results[2], 'Balanced Accuracy': results[3],
                   'Sensitivity': results[4], 'Specificity': results[5], 'Precision': results[6], 'Recall': results[7],
                   'F1-score': results[8]}

    print('\n')
    key_max_len = 16
    val_max_len = 10

    tag = '--'
    if ind is False:
        header = 'Final results of cross validation'
    else:
        header = 'Results of independent test'
    header_str1 = '+' + tag.center(key_max_len + val_max_len + 9, '-') + '+'
    header_str2 = '|' + header.center(key_max_len + val_max_len + 9, ' ') + '|'
    print(header_str1)
    print(header_str2)

    up_dn_str = '+' + tag.center(key_max_len + 4, '-') + '+' + tag.center(val_max_len + 4, '-') + '+'
    print(up_dn_str)
    for key, val in metric_dict.items():
        var_str = '%.4f' % val
        temp_str = '|' + str(key).center(key_max_len + 4, ' ') + '|' + var_str.center(val_max_len + 4, ' ') + '|'
        print(temp_str)
    print(up_dn_str)

    print('\n')


def mll_print_metric_dict(results, ind):
    metric_dict = {'hamming_loss': results[0], 'accuracy': results[1], 'jaccard_similarity': results[2],
                   'precision': results[3], 'recall': results[4], 'f1_score': results[5]}

    print('\n')
    key_max_len = 16
    val_max_len = 10

    tag = '--'
    if ind is False:
        header = 'Final results of cross validation'
    else:
        header = 'Results of independent test'
    header_str1 = '+' + tag.center(key_max_len + val_max_len + 9, '-') + '+'
    header_str2 = '|' + header.center(key_max_len + val_max_len + 9, ' ') + '|'
    print(header_str1)
    print(header_str2)

    up_dn_str = '+' + tag.center(key_max_len + 4, '-') + '+' + tag.center(val_max_len + 4, '-') + '+'
    print(up_dn_str)
    for key, val in metric_dict.items():
        var_str = '%.4f' % val
        temp_str = '|' + str(key).center(key_max_len + 4, ' ') + '|' + var_str.center(val_max_len + 4, ' ') + '|'
        print(temp_str)
    print(up_dn_str)

    print('\n')


def final_results_output(results, out_path, ind=False, multi=False):
    if multi is True:
        acc = float(results)
        acc_re = 'Acc = %.4f' % acc
        eval_re = [acc_re]
    else:
        acc_re = 'Acc = %.4f' % results[0]
        mcc_re = 'MCC = %.4f' % results[1]
        auc_re = 'AUC = %.4f' % results[2]
        bcc_re = 'BAcc = %.4f' % results[3]
        sn_re = 'Sn = %.4f' % results[4]
        sp_re = 'Sp = %.4f' % results[5]
        p_re = 'Precision = %.4f' % results[6]
        r_re = 'Recall = %.4f' % results[7]
        f1_re = 'F1 = %.4f\n' % results[8]
        eval_re = [acc_re, mcc_re, auc_re, bcc_re, sn_re, sp_re, p_re, r_re, f1_re]

    if ind is True:
        filename = out_path + 'ind_final_results.txt'
    else:
        filename = out_path + 'final_results.txt'
    with open(filename, 'w') as f:
        f.write('The final results of cross validation are as follows:\n')
        for i in eval_re:
            f.write(i)
            f.write("\n")
    full_path = os.path.abspath(filename)
    if os.path.isfile(full_path):
        print('The output file for final results can be found:')
        print(full_path)
        print('\n')


def mll_final_results_output(results, out_path, ind=False):
    hamming_loss = 'Hamming loss = %.4f' % results[0]
    accuracy = 'Acc = %.4f' % results[1]
    jaccard_similarity = 'Jaccard similarity = %.4f' % results[2]
    precision = 'Precision = %.4f' % results[3]
    recall = 'Recall = %.4f' % results[4]
    f1_score = 'F1 = %.4f\n' % results[5]

    eval_re = [hamming_loss, accuracy, jaccard_similarity, precision, recall, f1_score]

    if ind is True:
        filename = out_path + 'ind_final_results.txt'
    else:
        filename = out_path + 'final_results.txt'

    with open(filename, 'w') as f:
        f.write('The final results of cross validation are as follows:\n')
        for i in eval_re:
            f.write(i)
            f.write("\n")

    full_path = os.path.abspath(filename)
    if os.path.isfile(full_path):
        print('The output file for final results can be found:')
        print(full_path)
        print('\n')


def prob_output(true_labels, predicted_labels, prob_list, out_path, ind=False):
    prob_file = out_path + "prob_out.txt"
    if ind is True:
        prob_file = out_path + "ind_prob_out.txt"
    with open(prob_file, 'w') as f:
        head = 'Sample index' + '\t' + 'True labels' + '\t' + 'predicted labels' + '\t' + 'probability values' + '\n'
        f.write(head)
        for i, (k, m, n) in enumerate(zip(true_labels, predicted_labels, prob_list)):
            line = str(i + 1) + '\t' + str(k) + '\t' + str(m) + '\t' + str(n) + '\n'
            f.write(line)
    full_path = os.path.abspath(prob_file)
    if os.path.isfile(full_path):
        print('The output file for probability values can be found:')
        print(full_path)
        print('\n')


def mll_prob_output(true_labels, predicted_labels, prob_list, out_path, ind=False):
    if issparse(true_labels):
        true_labels = true_labels.toarray()

    if issparse(predicted_labels):
        predicted_labels = predicted_labels.toarray()
    predicted_labels = predicted_labels.astype(np.int, copy=True)

    assert true_labels.shape == predicted_labels.shape, \
        'err! with true_labels of shape {} while predicted_labels of shape {}'.format(
            true_labels.shape, predicted_labels.shape)

    if prob_list is None:
        prob_list = np.zeros(true_labels.shape, np.int)

    prob_file = out_path + "prob_out.csv"
    if ind is True:
        prob_file = out_path + "ind_prob_out.csv"

    len_label = true_labels.shape[1]*2
    header = ('Sample index',
              'True labels'.center(len_label, ' '),
              'predicted labels'.center(len_label, ' '),
              'probability values'.center(len_label*2, ' '))

    with open(prob_file, 'w', encoding='utf-8', newline='') as f:
        out = csv.writer(f)  # 创建writer对象
        out.writerow(header)
        for i, (k, m, n) in enumerate(zip(true_labels,
                                          predicted_labels,
                                          prob_list)):
            nn = tuple(map(lambda x: round(x, 4), n))
            line = tuple((i+1, tuple(k), tuple(m), nn))
            out.writerow(line)

    full_path = os.path.abspath(prob_file)
    if os.path.isfile(full_path):
        print('The output file for probability values can be found:')
        print(full_path)
        print('\n')
        

def prob_output_res(true_labels, predicted_labels, prob_list, out_path, ind=False):
    prob_file = out_path + "probability_values.txt"
    if ind is True:
        prob_file = out_path + "Ind_probability_values.txt"
    with open(prob_file, 'w') as f:
        for i in range(len(true_labels)):
            for k, m, n in zip(true_labels[i], predicted_labels[i], prob_list[i]):
                f.write(str(k))
                f.write('\t')
                f.write(str(m))
                f.write('\t')
                f.write(str(n))
                f.write('\n')
            f.write(' ' + '\n')
        f.close()
    full_path = os.path.abspath(prob_file)
    if os.path.isfile(full_path):
        print('The output file for probability values can be found:')
        print(full_path)
