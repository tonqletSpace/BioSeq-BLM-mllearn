import random

import numpy as np
import torch
from sklearn import preprocessing
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted
from skmultilearn.problem_transform import LabelPowerset
from skorch import NeuralNetClassifier
from torch import nn, optim
from scipy.sparse import issparse, lil_matrix, csr_matrix, csc_matrix

from ..utils.utils_net import TorchNetSeq, FORMER, MllBaseTorchNetSeq, sequence_mask, TrmDataset
from ..utils.utils_plot import plot_roc_curve, plot_pr_curve, plot_roc_ind, plot_pr_ind
from ..utils.utils_results import performance, final_results_output, prob_output, print_metric_dict, mll_performance, \
    mll_final_results_output, mll_prob_output, mll_print_metric_dict
from ..utils.utils_mll import BLMLabelPowerset, MllDeepNetSeq, get_mll_deep_model, get_lp_num_class, \
    mll_result_sparse_check, Mll_ENSEMBLE_Methods, is_mll_ensemble_methods, mll_hyper_param_show


def get_partition(feature, target, length, train_index, val_index):
    x_train = feature[train_index]
    x_val = feature[val_index]
    y_train = target[train_index]
    y_val = target[val_index]
    train_length = np.array(length)[train_index]
    test_length = np.array(length)[val_index]

    return x_train, x_val, y_train, y_val, train_length, test_length


def dl_cv_process(ml, vectors, labels, seq_length_list, max_len, folds, out_dir, params_dict):

    results = []
    cv_labels = []
    cv_prob = []

    predicted_labels = np.zeros(len(seq_length_list))
    predicted_prob = np.zeros(len(seq_length_list))

    count = 0
    criterion = nn.CrossEntropyLoss()
    in_dim = vectors.shape[-1]
    num_class = len(set(labels))
    multi = True if num_class > 2 else False
    for train_index, val_index in folds:
        x_train, x_val, y_train, y_val, train_length, test_length = get_partition(vectors, labels, seq_length_list,
                                                                                  train_index, val_index)
        model = TorchNetSeq(ml, max_len, criterion, params_dict).net_type(in_dim, num_class)
        optimizer = optim.Adam(model.parameters(), lr=params_dict['lr'])
        epochs = params_dict['epochs']
        # 筛选最后的模型参数
        min_loss = float('inf')
        final_predict_list = []
        final_target_list = []
        final_prob_list = []
        for epoch in range(1, epochs+1):
            TorchNetSeq(ml, max_len, criterion, params_dict).train(model, optimizer, x_train, y_train, train_length,
                                                                   epoch)
            predict_list, target_list, prob_list, test_loss = TorchNetSeq(ml, max_len, criterion, params_dict).\
                test(model, x_val, y_val, test_length)
            if test_loss < min_loss:
                min_loss = test_loss
                final_predict_list = predict_list
                final_target_list = target_list
                final_prob_list = prob_list

        result = performance(final_target_list, final_predict_list, final_prob_list, multi, res=True)
        results.append(result)

        cv_labels.append(final_target_list)
        cv_prob.append(final_prob_list)

        # 这里为保存概率文件准备
        predicted_labels[val_index] = np.array(final_predict_list)
        predicted_prob[val_index] = np.array(final_prob_list)
        count += 1
        print("Round[%d]: Accuracy = %.3f" % (count, result[0]))
    print('\n')
    plot_roc_curve(cv_labels, cv_prob, out_dir)  # 绘制ROC曲线
    plot_pr_curve(cv_labels, cv_prob, out_dir)  # 绘制PR曲线

    final_results = np.array(results).mean(axis=0)
    print_metric_dict(final_results, ind=False)

    final_results_output(final_results, out_dir, ind=False, multi=multi)  # 将指标写入文件
    prob_output(labels, predicted_labels, predicted_prob, out_dir)  # 将标签对应概率写入文件


def get_output_space_dim(y, mll, params_dict):
    if mll in ['BR']:  # binary classification
        return 2

    if mll in Mll_ENSEMBLE_Methods:
        return None

    if mll in ['LP']:
        return get_lp_num_class(y)


def mll_dl_cv_process(need_marginal_data, mll, ml, vectors, embed_size,
                      labels, seq_length_list, max_len, folds, out_dir, params_dict):

    mll_hyper_param_show(mll, ml, params_dict, is_optimal=True, print_len=60)

    tmp_shape = labels.get_shape()
    if need_marginal_data:
        tmp_shape = (tmp_shape[0]-2, tmp_shape[1])
    predicted_labels = np.zeros(tmp_shape, dtype=np.int32)  # (N, q)
    predicted_prob = np.zeros(tmp_shape)  # (N, q)

    results = []
    count = 0
    for train_index, val_index in folds:
        x_train, x_val, y_train, y_val, train_length, test_length = get_partition(
            vectors, labels, seq_length_list, train_index, val_index)
        num_class = get_output_space_dim(y_train, mll, params_dict)
        lp_args = ml, max_len, embed_size, params_dict
        mll_clf = get_mll_deep_model(num_class, mll, *lp_args)

        # blm是每个epoch都测试，选最好的测试结果；这里用fit后的结果来预测
        final_predict_list, final_prob_list = do_dl_fit_predict(
            mll, ml, mll_clf, x_train, y_train, train_length, max_len, x_val, test_length, 'LP', *lp_args)

        result = mll_performance(y_val, final_predict_list)
        results.append(result)

        # 这里为保存概率文件准备
        predicted_labels[val_index] = final_predict_list.toarray()

        if final_prob_list is not None:
            predicted_prob[val_index] = final_prob_list.toarray()

        count += 1
        print("Round[%d]: Accuracy = %.3f" % (count, result[0]))

    print('\n')
    final_results = np.array(results).mean(axis=0)
    mll_print_metric_dict(final_results, ind=False)
    mll_final_results_output(final_results, out_dir, ind=False)  # 将指标写入文件
    mll_prob_output(labels, predicted_labels, predicted_prob, out_dir)  # 将标签对应概率写入文件


def do_dl_fit_predict(mll, ml, mll_clf, x_train, y_train, train_length, max_len, x_val, test_length, *lp_args):
    mll_clf = do_dl_fit(mll, ml, mll_clf, x_train, y_train, train_length, max_len, *lp_args)
    final_predict_list, final_prob_list = do_dl_predict(mll, ml, mll_clf, max_len, x_val, test_length)

    return final_predict_list, final_prob_list


def do_dl_fit(mll, ml, mll_clf, x_train, y_train, train_length, max_len, *lp_args):
    if mll in Mll_ENSEMBLE_Methods:  # ensemble
        if ml in FORMER:
            # 额外参数
            mll_clf.fit(TrmDataset(x_train, y_train, train_length, max_len), None, *lp_args)
        else:
            # 额外参数
            mll_clf.fit(x_train, y_train, *lp_args)
    else:
        if ml in FORMER:
            mll_clf.fit(TrmDataset(x_train, y_train, train_length, max_len))
        else:
            mll_clf.fit(x_train, y_train)

    return mll_clf


def do_dl_predict(mll, ml, mll_clf, max_len, x_val, test_length):
    if mll in Mll_ENSEMBLE_Methods:  # ensemble
        if ml in FORMER:
            final_predict_list = mll_clf.predict(TrmDataset(x_val, None, test_length, max_len))  # (N, q
            final_prob_list = None
        else:
            final_predict_list = mll_clf.predict(x_val)  # (N, q
            final_prob_list = None
    else:
        if ml in FORMER:
            final_predict_list = mll_clf.predict(TrmDataset(x_val, None, test_length, max_len))  # (N, q
            final_prob_list = mll_clf.predict_proba(TrmDataset(x_val, None, test_length, max_len))  # (N, n
        else:
            final_predict_list = mll_clf.predict(x_val)  # (N, q
            final_prob_list = mll_clf.predict_proba(x_val)  # (N, n

    return final_predict_list, final_prob_list


def do_ml_fit_predict(mll, ml, mll_clf, x_train, y_train, x_val, params_dict):
    do_ml_fit(mll, ml, mll_clf, x_train, y_train, params_dict)
    y_val_ = mll_result_sparse_check(mll, mll_clf.predict(x_val))

    return y_val_


def do_ml_fit(mll, ml, mll_clf, x_train, y_train, params_dict):
    if is_mll_ensemble_methods(mll):  # ensemble
        mll_clf.fit(x_train, y_train, 'LP', ml, params_dict=params_dict)
    else:
        mll_clf.fit(x_train, y_train)


def dl_ind_process(ml, vectors, labels, seq_length_list, ind_vectors, ind_labels, ind_seq_length_list, max_len, out_dir,
                   params_dict):
    criterion = nn.CrossEntropyLoss()
    in_dim = vectors.shape[-1]
    num_class = len(set(labels))
    multi = True if num_class > 2 else False
    model = TorchNetSeq(ml, max_len, criterion, params_dict).net_type(in_dim, num_class)
    optimizer = optim.Adam(model.parameters(), lr=params_dict['lr'])
    epochs = params_dict['epochs']
    # 筛选最后的模型参数
    min_loss = float('inf')
    final_predict_list = []
    final_target_list = []
    final_prob_list = []

    for epoch in range(1, epochs+1):
        TorchNetSeq(ml, max_len, criterion, params_dict).train(model, optimizer, vectors, labels, seq_length_list,
                                                               epoch)
        predict_list, target_list, prob_list, test_loss = TorchNetSeq(ml, max_len, criterion, params_dict).\
            test(model, ind_vectors, ind_labels, ind_seq_length_list)
        if test_loss < min_loss:
            min_loss = test_loss
            final_predict_list = predict_list
            final_target_list = target_list
            final_prob_list = prob_list

    final_result = performance(final_target_list, final_predict_list, final_prob_list, multi, res=True)
    print_metric_dict(final_result, ind=True)

    plot_roc_ind(final_target_list, final_prob_list, out_dir)  # 绘制ROC曲线
    plot_pr_ind(final_target_list, final_prob_list, out_dir)  # 绘制PR曲线

    final_results_output(final_result, out_dir, ind=True, multi=multi)  # 将指标写入文件
    prob_output(final_target_list, final_predict_list, final_prob_list, out_dir)


def mll_dl_ind_process(need_marginal_data, mll, ml, vectors, labels, fixed_seq_len_list,
                       ind_vectors, ind_label_array, ind_fixed_seq_len_list,
                       embed_size, max_len, out_dir, params_dict):
    """
    用于训练的数据: vectors, fixed_seq_len_list, labels
    用于预测的独立测试集数据: ind_vectors, ind_fixed_seq_len_list
    用于评估的标签: ind_label_array
    共用的: embed_size, max_len
    out_dir: 独立测试输出文件
    """

    print("need_marginal_data", need_marginal_data)

    print("vectors.shape", vectors.shape)
    print("labels.shape", labels.shape)
    print("train_seq_length_list.shape", len(fixed_seq_len_list))
    print(type(fixed_seq_len_list))

    print("ind_vectors.shape", ind_vectors.shape)
    print("ind_labels.shape", ind_label_array.shape)
    print("ind_seq_length_list.shape", len(ind_fixed_seq_len_list))
    print(type(ind_fixed_seq_len_list))

    # exit()
    # blm是每个epoch都测试，选最好的测试结果；这里用fit后的结果来预测
    num_class = get_output_space_dim(labels, mll, params_dict)
    lp_args = ml, max_len, embed_size, params_dict
    mll_clf = get_mll_deep_model(num_class, mll, *lp_args)

    range_list = list(range(len(vectors)))
    random.shuffle(range_list)
    vectors, labels, train_length = vectors[range_list], labels[range_list], fixed_seq_len_list[range_list]

    mll_clf = do_dl_fit(mll, ml, mll_clf, vectors, labels, train_length, max_len, *lp_args)
    predict_list, prob_list = do_dl_predict(mll, ml, mll_clf, max_len, ind_vectors, ind_fixed_seq_len_list)

    final_predict_list = predict_list.toarray()

    if prob_list is not None:
        final_prob_list = prob_list.toarray()

    final_result = mll_performance(ind_label_array, final_predict_list)

    mll_print_metric_dict(final_result, ind=True)
    mll_final_results_output(final_result, out_dir, ind=True)  # 将指标写入文件
    mll_prob_output(ind_label_array, final_predict_list, final_prob_list, out_dir)  # 将标签对应概率写入文件
