from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import numpy as np

# from ..utils.utils_results import performance, final_results_output, prob_output, print_metric_dict
# from ..utils.utils_plot import plot_roc_curve, plot_pr_curve, plot_roc_ind, plot_pr_ind
# from ..utils.utils_math import sampling
# from ..utils.utils_read import FormatRead

Metric_List = ['Acc', 'MCC', 'AUC', 'BAcc', 'Sn', 'Sp', 'Pr', 'Rc', 'F1']


def mll_ml_cv_process(ml, vectors, labels, folds, metric, sp, multi, res, params_dict):
    results = []

    print_len = 40
    if ml == 'SVM':
        temp_str1 = '  cost = 2 ** ' + str(params_dict['cost']) + ' | ' + 'gamma = 2 ** ' + \
                    str(params_dict['gamma']) + '  '
    else:
        temp_str1 = '  tree = ' + str(params_dict['tree']) + '  '
    print(temp_str1.center(print_len, '+'))

    for train_index, val_index in folds:
        x_train, y_train, x_val, y_val = get_partition(vectors, labels, train_index, val_index)
        if sp != 'none':
            x_train, y_train = sampling(sp, x_train, y_train)
        if ml == 'SVM':
            clf = svm.SVC(C=2 ** params_dict['cost'], gamma=2 ** params_dict['gamma'], probability=True)
        else:
            clf = RandomForestClassifier(random_state=42, n_estimators=params_dict['tree'])
        clf.fit(x_train, y_train)
        y_val_prob = clf.predict_proba(x_val)[:, 1]  # 这里为什么是1呢

        y_val_ = clf.predict(x_val)

        result = performance(y_val, y_val_, y_val_prob, multi, res)
        # acc, mcc, auc, balance_acc, sn, sp, p, r, f1
        results.append(result)

    cv_results = np.array(results).mean(axis=0) if not multi else [np.array(results).mean(axis=0)]

    # print('metric: ', metric)
    # print('cv_results: ', cv_results)
    params_dict['metric'] = cv_results[metric]
    # print('params_dict: ', params_dict)
    temp_str2 = '  metric value: ' + Metric_List[metric] + ' = ' + '%.3f  ' % cv_results[metric]
    print(temp_str2.center(print_len, '*'))
    # print('\n')
    # return False
    # exit()
    return params_dict
