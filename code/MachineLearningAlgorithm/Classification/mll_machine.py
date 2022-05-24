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

        clf = get_mll_ml_model(mll, ml, params_dict)

        clf.fit(x_train, y_train)

        # print('fs:')
        # print(train_index)
        # print(val_index)
        # for c in clf.classifiers_:
        #     print(c.classes_)

        y_val_ = clf.predict(x_val)

        # 'Ham', 'Acc', 'Jac', 'Pr', 'Rc', 'F1'
        result = mll_performance(y_val, y_val_)
        results.append(result)

    # exit()
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


def get_mll_ml_model(mll, ml, params_dict):
    if mll == 'BR':
        return BinaryRelevance(classifier=get_ml_model(ml, params_dict), require_dense=[False, True])
    elif mll == 'CC':
        return ClassifierChain(classifier=get_ml_model(ml, params_dict), require_dense=[False, True])
    elif mll == 'LP':
        return LabelPowerset(classifier=get_ml_model(ml, params_dict), require_dense=[False, True])
    else:
        raise ValueError('mll method err')


def get_ml_model(ml, params_dict):
    if ml == 'SVM':
        return svm.SVC(C=2 ** params_dict['cost'], gamma=2 ** params_dict['gamma'], probability=True)
    elif ml == 'RF':
        return RandomForestClassifier(random_state=42, n_estimators=params_dict['tree'])
    else:
        raise ValueError('ml method err')


# def get_label_inducing_data(vectors):
#     e, q = vectors.shape[1], 1
#     aux_data, aux_label = lil_matrix((2, e)), lil_matrix((2, q))
#     return aux_data, [0, 1]


# def mll_validate_data(label_arrays):
    # for q dim label, assure the length of each dim is 2
