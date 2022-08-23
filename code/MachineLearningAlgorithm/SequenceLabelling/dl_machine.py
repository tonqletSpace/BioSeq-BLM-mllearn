import numpy as np
from torch import nn, optim

from ..utils.utils_net import TorchNetRes, criterion_func
# from ..utils.utils_plot import plot_roc_curve, plot_pr_curve, plot_roc_ind, plot_pr_ind
from ..utils.utils_results import performance, final_results_output, prob_output, print_metric_dict


def get_partition(feature, target, length, train_index, val_index):
    feature = np.array(feature)
    x_train = feature[train_index]
    x_val = feature[val_index]
    y_train = np.array(target)[train_index]
    y_val = np.array(target)[val_index]
    train_length = np.array(length)[train_index]
    test_length = np.array(length)[val_index]

    return x_train, x_val, y_train, y_val, train_length, test_length


def dl_cv_process(ml, vectors, labels, seq_length_list, max_len, folds, out_dir, params_dict):
    results = []
    cv_labels = []
    cv_prob = []

    all_true_labels = []
    all_pre_labels = []
    all_prob = []

    # predicted_labels = np.zeros(len(seq_length_list))
    # predicted_prob = np.zeros(len(seq_length_list))
    print("vectors.shape", vectors.shape)
    print("labels.shape", labels.shape)
    # exit()
    count = 0
    criterion = criterion_func
    in_dim = vectors.shape[-1]
    print('in MachineLearningAlgorithm.SequenceLabelling.dl_machine\n', 'num_class=2 fixed')
    num_class = 2
    multi = True if num_class > 2 else False

    eval_cnt = 0
    for train_index, val_index in folds:
        x_train, x_val, y_train, y_val, train_length, test_length = get_partition(vectors, labels, seq_length_list,
                                                                                  train_index, val_index)
        print('dl without fagment'.center(100, '*'))
        print("x_train", x_train.shape)
        print("y_train", y_train.shape)
        print("x_val", x_val.shape)
        print("y_val", y_val.shape)
        # print(isinstance(y_val, list))
        # exit()
        model = TorchNetRes(ml, max_len, criterion, params_dict).net_type(in_dim, num_class)
        optimizer = optim.Adam(model.parameters(), lr=params_dict['lr'])
        epochs = params_dict['epochs']
        # 筛选最后的模型参数
        min_loss = float('inf')
        final_predict_list = []
        final_target_list = []
        final_prob_list = []
        # final_prob_list_format = []
        # final_predict_list_format = []
        for epoch in range(1, epochs+1):
            TorchNetRes(ml, max_len, criterion, params_dict).train(model, optimizer, x_train, y_train, train_length,
                                                                   epoch)
            predict_list, target_list, prob_list, test_loss, prob_list_format, predict_list_format = \
                TorchNetRes(ml, max_len, criterion, params_dict).test(model, x_val, y_val, test_length)
            if test_loss < min_loss:
                min_loss = test_loss
                final_predict_list = predict_list
                final_target_list = target_list
                final_prob_list = prob_list

        print("final_target_list", len(final_target_list))
        print("final_predict_list", len(final_predict_list))

        result = performance(final_target_list, final_predict_list, final_prob_list, multi, True)
        results.append(result)
        print("result", len(result))
        exit()

        cv_labels.append(final_target_list)
        cv_prob.append(final_predict_list)
        # 所有的真实标签，预测标签和预测概率
        all_true_labels.extend(final_target_list)
        all_pre_labels.extend(final_predict_list)
        all_prob.extend(final_predict_list)

        # 这里为保存概率文件准备
        count += 1
        print("Round[%d]: Accuracy = %.3f" % (count, result[0]))
    print('\n')
    # plot_roc_curve(cv_labels, cv_prob, out_dir)  # 绘制ROC曲线
    # plot_pr_curve(cv_labels, cv_prob, out_dir)  # 绘制PR曲线

    print("cv_labels", len(cv_labels))
    print("cv_prob", len(cv_prob))
    # print("result", len(result))
    final_results = np.array(results).mean(axis=0)
    # table_metric(final_results, True)
    print_metric_dict(final_results, ind=False)

    final_results_output(final_results, out_dir, ind=False, multi=multi)  # 将指标写入文件
    prob_output(all_true_labels, all_pre_labels, all_prob, out_dir)


def dl_ind_process(ml, vectors, labels, seq_length_list, ind_vectors, ind_labels, ind_seq_length_list, max_len, out_dir,
                   params_dict):
    criterion = nn.CrossEntropyLoss()
    in_dim = vectors.shape[-1]
    num_class = 2
    multi = True if num_class > 2 else False
    model = TorchNetRes(ml, max_len, criterion, params_dict).net_type(in_dim, num_class)
    optimizer = optim.Adam(model.parameters(), lr=params_dict['lr'])
    epochs = params_dict['epochs']
    # 筛选最后的模型参数
    min_loss = float('inf')
    final_predict_list = []
    final_target_list = []
    final_prob_list = []

    for epoch in range(1, epochs+1):
        TorchNetRes(ml, max_len, criterion, params_dict).train(model, optimizer, vectors, labels, seq_length_list,
                                                               epoch)
        predict_list, target_list, prob_list, test_loss, prob_list_format, predict_list_format = \
            TorchNetRes(ml, max_len, criterion, params_dict).test(model, ind_vectors, ind_labels, ind_seq_length_list)
        if test_loss < min_loss:
            min_loss = test_loss
            final_predict_list = predict_list
            final_target_list = target_list
            final_prob_list = prob_list

    final_result = performance(final_target_list, final_predict_list, final_prob_list, multi, True)
    # table_metric(final_result, True)
    print_metric_dict(final_result, ind=True)

    # plot_roc_ind(final_target_list, final_prob_list, out_dir)  # 绘制ROC曲线
    # plot_pr_ind(final_target_list, final_prob_list, out_dir)  # 绘制PR曲线

    final_results_output(final_result, out_dir, ind=True, multi=multi)  # 将指标写入文件
    # prob_output_res(final_target_list, final_predict_list, final_prob_list, out_dir)
