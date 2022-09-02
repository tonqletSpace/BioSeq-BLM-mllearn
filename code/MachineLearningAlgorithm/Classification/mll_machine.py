from skmultilearn.adapt import MLARAM, BRkNNaClassifier, MLkNN, BRkNNbClassifier
from skmultilearn.ensemble import RakelO
from skmultilearn.problem_transform import BinaryRelevance, ClassifierChain, LabelPowerset
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import numpy as np
from skmultilearn.ext import Meka, download_meka
from scipy.sparse import lil_matrix, issparse

from .dl_machine import do_ml_fit_predict
from ..utils.utils_mll import BLMRAkELo, BLMMeka, mll_sparse_check, get_mll_ml_model
from ..utils.utils_results import mll_performance

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

Metric_List = ['Ham', 'Acc', 'Jac', 'Pr', 'Rc', 'F1']


def mll_ml_cv_process(mll, ml, vectors, labels, folds, metric, params_dict):
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
        x_train, y_train, x_val, y_val = mll_sparse_check(mll, *get_partition(vectors, labels, train_index, val_index))

        # if sp != 'none':
        #     x_train, y_train = sampling(sp, x_train, y_train)
        clf = get_mll_ml_model(mll, ml, params_dict)
        y_val_ = do_ml_fit_predict(mll, ml, clf, x_train, y_train, x_val, params_dict)

        result = mll_performance(y_val, y_val_)
        results.append(result)

    cv_results = np.array(results).mean(axis=0)

    params_dict['metric'] = cv_results[metric]
    temp_str2 = '  metric value: ' + Metric_List[metric] + ' = ' + '%.3f  ' % cv_results[metric]
    print(temp_str2.center(print_len, '*'))
    return params_dict


def get_partition(vectors, labels, train_index, val_index):
    x_train = vectors[train_index]
    x_val = vectors[val_index]
    y_train = labels[train_index]
    y_val = labels[val_index]

    return x_train, y_train, x_val, y_val

