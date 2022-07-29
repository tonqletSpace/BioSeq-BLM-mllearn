from skmultilearn.adapt import MLARAM, BRkNNaClassifier, MLkNN, BRkNNbClassifier
from skmultilearn.problem_transform import BinaryRelevance, ClassifierChain, LabelPowerset
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import numpy as np
from skmultilearn.ext import Meka

# from ..utils.utils_results import performance, final_results_output, prob_output, print_metric_dict
# from ..utils.utils_plot import plot_roc_curve, plot_pr_curve, plot_roc_ind, plot_pr_ind
# from ..utils.utils_math import sampling
# from ..utils.utils_read import FormatRead
from scipy.sparse import lil_matrix, issparse

from ..utils.utils_mll import is_mll_instance_methods
from ..utils.utils_results import mll_performance

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

Metric_List = ['Ham', 'Acc', 'Jac', 'Pr', 'Rc', 'F1']


def mll_ml_cv_process(mll, ml, vectors, labels, folds, metric, params_dict):
    assert isinstance(vectors, lil_matrix), 'error'

    if ml:
        print_len = 40
        if ml == 'SVM':
            temp_str1 = '  cost = 2 ** ' + str(params_dict['cost']) + ' | ' + 'gamma = 2 ** ' + \
                        str(params_dict['gamma']) + '  '
        else:
            temp_str1 = '  tree = ' + str(params_dict['tree']) + '  '
        print(temp_str1.center(print_len, '+'))

    print_len = 40
    print(str(params_dict).center(print_len, '+'))

    results = []
    for train_index, val_index in folds:
        # x_train, y_train, x_val, y_val =
        x_train, y_train, x_val, y_val = mll_sparse_check(mll, *get_partition(vectors, labels, train_index, val_index))

        # if sp != 'none':
        #     x_train, y_train = sampling(sp, x_train, y_train)
        clf = get_mll_ml_model(mll, ml, params_dict)
        clf.fit(x_train, y_train)
        y_val_ = mll_result_sparse_check(mll, clf.predict(x_val))
        # 'Ham', 'Acc', 'Jac', 'Pr', 'Rc', 'F1'
        result = mll_performance(y_val, y_val_)
        results.append(result)

    cv_results = np.array(results).mean(axis=0)

    params_dict['metric'] = cv_results[metric]
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
    if ml:
        if mll == 'BR':
            return BinaryRelevance(classifier=ml_model_factory(ml, params_dict), require_dense=[True, True])
        elif mll == 'CC':
            return ClassifierChain(classifier=ml_model_factory(ml, params_dict), require_dense=[True, True])
        elif mll == 'LP':
            return LabelPowerset(classifier=ml_model_factory(ml, params_dict), require_dense=[True, True])
        else:
            raise ValueError('mll ml method err')

    # ml is None, no ml base clf
    return mll_model_factory(mll, params_dict)


def ml_model_factory(ml, params_dict):
    if ml == 'SVM':
        return svm.SVC(C=2 ** params_dict['cost'], gamma=2 ** params_dict['gamma'], probability=True)
    elif ml == 'RF':
        return RandomForestClassifier(random_state=42, n_estimators=params_dict['tree'])
    else:
        raise ValueError('ml method err')


def mll_model_factory(mll, params_dict):
    meka_classpath = '/Users/maiqi/mll/aux_pkgs/meka'
    if mll == 'MLkNN':
        return MLkNN(k=params_dict['mll_kNN_k'],
                     s=params_dict['MLkNN_s'],
                     ignore_first_neighbours=params_dict['MLkNN_ignore_first_neighbours'])
    elif mll == 'BRkNNaClassifier':
        return BRkNNaClassifier(k=params_dict['mll_kNN_k'])
    elif mll == 'BRkNNbClassifier':
        return BRkNNbClassifier(k=params_dict['mll_kNN_k'])
    elif mll == 'MLARAM':
        return MLARAM(vigilance=params_dict['MLARAM_vigilance'],
                      threshold=params_dict['MLARAM_threshold'],
                      neurons=params_dict['MLARAM_neurons'] if 'MLARAM_neurons' in params_dict.keys()
                      else None)
    elif mll == 'CLR':
        return Meka(
            meka_classifier="meka.classifiers.multilabel.BR",  # Binary Relevance
            weka_classifier="weka.classifiers.bayes.NaiveBayesMultinomial",  # with Naive Bayes single-label classifier
            meka_classpath=meka_classpath,  # obtained via download_meka
            java_command='/usr/bin/java'  # path to java executable
        )
    else:
        raise ValueError('mll method err')


def mll_sparse_check(mll, *data_list):
    if mll in ['MLARAM']:
        # 2 usage of this transformation
        # x_train, y_train, x_val, y_va)
        # x_train, y_train
        data_list = [e.tocsc() for e in data_list]

    if len(data_list) == 2:
        return data_list[0], data_list[1]
    if len(data_list) == 4:
        return data_list[0], data_list[1], data_list[2], data_list[3]


def mll_result_sparse_check(mll, res):
    if mll in ['MLARAM'] and not issparse(res):
        return lil_matrix(res)
    return res


def mll_ml_fit(mll, clf, x_train, y_train):
    if is_mll_instance_methods(mll):
        print(" in ".center(100, '*'))
        clf.fit(x_train, y_train)
    else:
        clf.fit(x_train, y_train)


# def get_label_inducing_data(vectors):
#     e, q = vectors.shape[1], 1
#     aux_data, aux_label = lil_matrix((2, e)), lil_matrix((2, q))
#     return aux_data, [0, 1]


# def mll_validate_data(label_arrays):
    # for q dim label, assure the length of each dim is 2
