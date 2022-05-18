from skmultilearn.problem_transform import BinaryRelevance, ClassifierChain, LabelPowerset
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import numpy as np

# from ..utils.utils_results import performance, final_results_output, prob_output, print_metric_dict
# from ..utils.utils_plot import plot_roc_curve, plot_pr_curve, plot_roc_ind, plot_pr_ind
# from ..utils.utils_math import sampling
# from ..utils.utils_read import FormatRead
from scipy.sparse import lil_matrix

from ..utils.utils_results import mll_performance

Metric_List = ['Ham', 'Acc', 'Jac', 'Pr', 'Rc', 'F1']


def mll_ml_cv_process(mll, ml, vectors, labels, folds, metric, params_dict):
    assert isinstance(vectors, lil_matrix), 'error'

    print_len = 40
    if ml == 'SVM':
        temp_str1 = '  cost = 2 ** ' + str(params_dict['cost']) + ' | ' + 'gamma = 2 ** ' + \
                    str(params_dict['gamma']) + '  '
    else:
        temp_str1 = '  tree = ' + str(params_dict['tree']) + '  '
    print(temp_str1.center(print_len, '+'))

    results = []
    for train_index, val_index in folds:
        x_train, y_train, x_val, y_val = get_partition(vectors, labels, train_index, val_index)
        # if sp != 'none':
        #     x_train, y_train = sampling(sp, x_train, y_train)

        if mll == 'BR':
            if ml == 'SVM':
                clf = BinaryRelevance(
                    classifier=svm.SVC(C=2 ** params_dict['cost'], gamma=2 ** params_dict['gamma']),
                    require_dense=[False, True]
                )
            else:
                clf = BinaryRelevance(
                    classifier=RandomForestClassifier(random_state=42, n_estimators=params_dict['tree']),
                    require_dense=[False, True]
                )
        elif mll == 'CC':
            if ml == 'SVM':
                clf = ClassifierChain(
                    classifier=svm.SVC(C=2 ** params_dict['cost'], gamma=2 ** params_dict['gamma']),
                    require_dense=[False, True]
                )
            else:
                clf = ClassifierChain(
                    classifier=RandomForestClassifier(random_state=42, n_estimators=params_dict['tree']),
                    require_dense=[False, True]
                )
        elif mll == 'LP':
            if ml == 'SVM':
                clf = LabelPowerset(
                    classifier=svm.SVC(C=2 ** params_dict['cost'], gamma=2 ** params_dict['gamma']),
                    require_dense=[False, True]
                )
            else:
                clf = LabelPowerset(
                    classifier=RandomForestClassifier(random_state=42, n_estimators=params_dict['tree']),
                    require_dense=[False, True]
                )
        else:
            raise ValueError('mll method err')

        clf.fit(x_train, y_train)
        y_val_ = clf.predict(x_val)

        # 'Ham', 'Acc', 'Jac', 'Pr', 'Rc', 'F1'
        result = mll_performance(y_val, y_val_)
        results.append(result)

    cv_results = np.array(results).mean(axis=0)

    # print('metric: ', metric)
    # print('cv_results: ', cv_results)
    params_dict['metric'] = cv_results[metric]  # this is metric_index
    # print('params_dict: ', params_dict)
    temp_str2 = '  metric value: ' + Metric_List[metric] + ' = ' + '%.3f  ' % cv_results[metric]
    print(temp_str2.center(print_len, '*'))
    # print('\n')
    # return False
    # exit()
    return params_dict


def get_partition(vectors, labels, train_index, val_index):
    x_train = vectors[train_index]
    x_val = vectors[val_index]
    y_train = labels[train_index]
    y_val = labels[val_index]

    return x_train, y_train, x_val, y_val