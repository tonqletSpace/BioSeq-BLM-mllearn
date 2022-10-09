import multiprocessing
import os
import time

from CheckAll import check_contain_chinese, ml_params_check, make_params_dicts, \
    res_feature_check, Machine_Learning_Algorithm, DeepLearning, dl_params_check, \
    Method_Res, mll_prepare4train_res, \
    mll_prepare4train_seq, mode_params_check, mll_seq_sys_check, mll_params_check, mll_ensemble_check, \
    mll_params_select
from FeatureAnalysis import fa_process
from FeatureExtractionMode.OHE.OHE4vec import ohe2res_base, \
    mll_sliding_win2files
from FeatureExtractionMode.utils.utils_write import fixed_len_control, mll_gen_label_matrix, \
    mll_read_res_seq_file, mll_out_res_file
from MachineLearningAlgorithm.Classification.dl_machine import mll_dl_cv_process, \
    mll_dl_ind_process
from MachineLearningAlgorithm.Classification.ml_machine import mll_ml_ind_results
from MachineLearningAlgorithm.utils.utils_read import read_base_vec_list4res, \
    mll_read_dl_data4res, \
    mll_files2vectors_seq
from MachineLearningRes import mll_one_cl_process
from FeatureExtractionRes import create_results_dir
from MachineLearningSeq import mll_ml_results
from MachineLearningAlgorithm.utils.utils_mll import mll_arg_parser, mll_meka_check


def mll_res_dl_fe_process(args, label_array, out_files):
    # 从out_files中读出 滑窗处理的 特征
    vectors = mll_files2vectors_seq(args, out_files, args.format, is_dl_flow=True)
    # 进一步处理特征，得到deep learning compatible的特征
    vectors, embed_size, fixed_seq_len_list = mll_read_dl_data4res(args, vectors, args.window)

    # 深度学习的独立测试和交叉验证分开
    if args.ind_seq_file is None:
        # 在参数便利前进行一系列准备工作: 1. 固定划分；2.设定指标；3.指定任务类型

        args = mll_prepare4train_seq(args, label_array, dl=True)
        # 构建深度学习分类器
        # fixed_len为args.window所替代
        mll_dl_cv_process(args.mll, args.ml, vectors, embed_size,
                          label_array, fixed_seq_len_list, args.window,
                          args.folds, args.results_dir, args.params_dict_list)
    else:
        # 独立验证开始
        mll_res_ind_dl_fe_process(args, vectors, label_array, fixed_seq_len_list,
                                  args.window, args.params_dict_list)


def mll_res_ml_fe_process(args, label_array, out_files):
    # 读取特征向量文件
    vectors = mll_files2vectors_seq(args, out_files, args.format)

    # 在参数便利前进行一系列准备工作: 1. 固定划分；2.设定指标；3.指定任务类型
    args = mll_prepare4train_res(args, label_array, dl=False)

    # ** 通过遍历SVM/RF参数字典列表来筛选参数 ** #
    # SVM/RF参数字典
    params_dict_list = args.params_dict_list
    # 多进程控制
    pool = multiprocessing.Pool(args.cpu)
    params_dict_list_pro = []
    print('\nParameter Selection Processing...\n')
    for i in range(len(params_dict_list)):
        params_dict = params_dict_list[i]
        mll_ensemble_check(label_array.shape[1], params_dict)
        mll_meka_check(args, params_dict)

        # TODO vectors 有问题！没有筛选？？？
        params_dict_list_pro.append(pool.apply_async(mll_one_cl_process,
                                                     (args, vectors, label_array, args.folds, params_dict)))

    pool.close()
    pool.join()
    # ** 筛选结束 ** #

    # 根据指标进行参数选择
    # TODO 筛选有问题
    params_selected = mll_params_select(params_dict_list_pro, args.results_dir)

    # 特征分析
    print(' Shape of Feature vectors: [%d, %d] '.center(66, '*') % (vectors.shape[0], vectors.shape[1]))
    print('\n')
    # if args.score == 'none':
    #     vectors = fa_process(args, vectors, label_array, after_ps=True)
    #     print(' Shape of Feature vectors after FA process: [%d, %d] '.center(66, '*') % (vectors.shape[0],
    #                                                                                      vectors.shape[1]))

    # 构建分类器
    model_path = mll_ml_results(args, vectors, label_array, args.folds, params_selected)
    # -------- 独立测试-------- #
    # 即，将独立测试数据集在最优的model上进行测试
    if args.ind_seq_file is not None:
        mll_res_ind_ml_fe_process(args, model_path, params_selected)


def mll_res_ind_dl_fe_process(args, vectors, label_array, fixed_seq_len_list, fixed_len, params_dict):
    print('########################## Independent Test Begin ##########################\n')
    ind_label_array, ind_out_files = mll_res_preprocess(args, is_ind=True)

    ind_vectors = mll_files2vectors_seq(args, ind_out_files, args.format, is_dl_flow=True)
    # fixed_len为args.window所替代, 问题转化
    ind_vectors, embed_size, ind_fixed_seq_len_list = mll_read_dl_data4res(args, ind_vectors, args.window)

    mll_dl_ind_process(args.mll, args.ml, vectors, label_array, fixed_seq_len_list,
                       ind_vectors, ind_label_array, ind_fixed_seq_len_list,
                       embed_size, fixed_len, args.results_dir, params_dict)
    print('########################## Independent Test Finish ##########################\n')


def mll_res_ind_ml_fe_process(args, model_path, params_selected):
    # TODO 为什么不复用？！因为要对独立数据做sliding_window，所以不能复用seq_level的流程
    # TODO 在全数据集上训练，再在ind数据上预测
    print('########################## Independent Test Begin ##########################\n')

    ind_label_array, ind_out_files = mll_res_preprocess(args, is_ind=True)

    # 读取独立测试数据集特征向量文件
    ind_vectors = mll_files2vectors_seq(args, ind_out_files, args.format)

    # 独立测试集特征分析
    print(' Shape of Ind Feature vectors: [%d, %d] '.center(66, '*') % (ind_vectors.shape[0], ind_vectors.shape[1]))
    print('\n')
    # if args.score == 'none':
    #     ind_vectors = fa_process(args, ind_vectors, ind_label_array, True, True)
    #     print(' Shape of Ind Feature vectors after FA process: [%d, %d] '.center(66, '*') % (ind_vectors.shape[0],
    #                                                                                          ind_vectors.shape[1]))
    # 构建独立测试集的分类器
    mll_ml_ind_results(args.mll, args.ml, ind_vectors, ind_label_array,
                       model_path, args.results_dir, params_selected)
    print('########################## Independent Test Finish ##########################\n')


def mll_res_preprocess(args, is_ind=False):
    """ 为独立测试步骤生成特征 """
    label_array = mll_generate_res_label_data(args, is_ind=is_ind)

    # 所有res特征在基准数据集上的基础输出文件
    if not is_ind:
        # TODO ?? 要加[mll]_csv吗？最开始怎么没有
        args.fea_file = args.results_dir + 'res_features[mll]_csv.txt'
        out_file = args.fea_file
    else:
        args.ind_fea_file = args.results_dir + 'ind_res_features[mll]_csv.txt'
        out_file = args.ind_fea_file

    mll_dump_res_data2file(args, out_file, is_ind=is_ind)

    return label_array, out_file


def mll_generate_res_label_data(args, is_ind=False):
    # 读取序列文件里每条序列的长度
    if not is_ind:
        seq_len_list, res_label_list = mll_read_res_seq_file(args.seq_file, args.label_file, args.category)
    else:
        seq_len_list, res_label_list = mll_read_res_seq_file(args.ind_seq_file, args.ind_label_file, args.category)

    # 控制序列的固定长度(只需要在benchmark dataset上操作一次）
    args.fixed_len = fixed_len_control(seq_len_list, args.fixed_len)
    label_array = mll_gen_label_matrix(res_label_list)

    return label_array


def mll_dump_res_data2file(args, fea_file, is_ind=False):
    # 所有res特征在基准数据集上的基础输出文件
    args.results_dir = create_results_dir(args, args.current_dir)
    # 提取残基层面特征,生成向量文件到fea_file
    ohe2res_base(args.seq_file, args.category, args.method, args.current_dir,
                 args.pp_file, args.rss_file, fea_file, args.cpu)

    # 读取base特征文件fea_file到内存
    vectors_list = read_base_vec_list4res(fea_file)

    # 划窗处理，创建并转存特征文件out_files
    out_files = mll_out_res_file(args.results_dir, args.format, ind=is_ind)
    assert args.window is not None, "please set window size!"
    mll_sliding_win2files(vectors_list, args.window, args.format, out_files)

    return out_files


def main(args):
    print("\nStep into analysis...\n")
    start_time = time.time()
    current_path = os.path.dirname(os.path.realpath(__file__))
    args.current_dir = os.path.dirname(os.getcwd())
    # 生成结果文件夹
    args.results_dir = create_results_dir(args, args.current_dir)

    # 判断中文目录
    check_contain_chinese(current_path)
    # 判断mode和ml的组合是否合理
    args.mode = 'OHE'
    args.score = 'none'
    mll_seq_sys_check(args, True)
    # 对每个残基层面的method进行检查
    res_feature_check(args)

    # # 读取序列文件里每条序列的长度
    label_array = mll_generate_res_label_data(args)

    # 对SVM或RF的参数进行检查
    all_params_list_dict = {}  # 包含了机器学习和特征提取的参数
    if args.ml in DeepLearning:
        all_params_list_dict = dl_params_check(args, all_params_list_dict)
        mll_params_check(args, all_params_list_dict)
        params_list_dict, all_params_list_dict = mode_params_check(args, all_params_list_dict)
        args.params_dict_list = make_params_dicts(all_params_list_dict)[0]

        mll_ensemble_check(label_array.shape[1], args.params_dict_list)
        mll_meka_check(args, args.params_dict_list)
    else:
        all_params_list_dict = ml_params_check(args, all_params_list_dict)
        mll_params_check(args, all_params_list_dict)
        args.params_dict_list = make_params_dicts(all_params_list_dict)

    # print('all_params_list_dict', all_params_list_dict)
    # print('args.params_dict_list', args.params_dict_list)

    fea_file = args.results_dir + 'res_features.txt'
    out_files = mll_dump_res_data2file(args, fea_file)

    # residue level 转化为 sequence level
    # label_array 对cv-process共用，对ind-process用做训练
    # vectors_list通过sliding window处理好，res_vectors2file存储在out_files文件里，在下面流程读出

    if args.ml in DeepLearning:
        mll_res_dl_fe_process(args, label_array, out_files)
    else:
        mll_res_ml_fe_process(args, label_array, out_files)

    print("Done.")
    print(("Used time: %.2fs" % (time.time() - start_time)))


if __name__ == '__main__':
    import argparse

    parse = argparse.ArgumentParser(prog='BioSeq-BLM-MLL', description="Step into analysis, please select parameters ")

    # parameters for whole framework
    parse.add_argument('-category', type=str, choices=['DNA', 'RNA', 'Protein'], required=True,
                       help="The category of input sequences.")

    parse.add_argument('-method', type=str, required=True, choices=Method_Res,
                       help="Please select feature extraction method for residue level analysis")
    # ----------------------- parameters for FeatureExtraction ---------------------- #
    # parameters for residue
    parse.add_argument('-window', type=int,
                       help="The sliding window technique will allocate every label a short sequence. "
                            "The window size equals to the length of short sequence.")

    # parameters for one-hot encoding
    parse.add_argument('-cpu', type=int, default=1,
                       help="The maximum number of CPU cores used for multiprocessing in generating frequency profile"
                            " or The number of CPU cores used for multiprocessing during parameter selection process "
                            "(default=1).")
    parse.add_argument('-pp_file', type=str,
                       help="The physicochemical properties file user input.\n"
                            "if input nothing, the default physicochemical properties is:\n"
                            "DNA dinucleotide: Rise, Roll, Shift, Slide, Tilt, Twist.\n"
                            "DNA trinucleotide: Dnase I, Bendability (DNAse).\n"
                            "RNA: Rise, Roll, Shift, Slide, Tilt, Twist.\n"
                            "Protein: Hydrophobicity, Hydrophilicity, Mass.")
    parse.add_argument('-rss_file', type=str,
                       help="The second structure file for all input sequences.(The order of a specific sequence "
                            "should be corresponding to the order in 'all_seq_file.txt' file")
    # ----------------------- parameters for feature analysis---------------------- #
    # standardization or normalization
    parse.add_argument('-sn', choices=['min-max-scale', 'standard-scale', 'L1-normalize', 'L2-normalize', 'none'],
                       default='none', help=" Choose method of standardization or normalization for feature vectors.")
    # clustering
    parse.add_argument('-cl', choices=['AP', 'DBSCAN', 'GMM', 'AGNES', 'Kmeans', 'none'], default='none',
                       help="Choose method for clustering.")
    parse.add_argument('-cm', default='sample', choices=['feature', 'sample'], help="The mode for clustering")
    parse.add_argument('-nc', type=int, help="The number of clusters.")
    # feature select
    parse.add_argument('-fs', choices=['chi2', 'F-value', 'MIC', 'RFE', 'Tree', 'none'], default='none',
                       help="Select feature select method. Please refer to sklearn feature selection module for more.")
    parse.add_argument('-nf', type=int, help="The number of features after feature selection.")
    # dimension reduction
    parse.add_argument('-dr', choices=['PCA', 'KernelPCA', 'TSVD', 'none'], default='none',
                       help="Choose method for dimension reduction.")
    parse.add_argument('-np', type=int, help="The dimension of main component after dimension reduction.")
    # rdb
    parse.add_argument('-rdb', choices=['no', 'fs', 'dr'], default='no',
                       help="Reduce dimension by:\n"
                            " 'no'---none;\n"
                            " 'fs'---apply feature selection to parameter selection procedure;\n"
                            " 'dr'---apply dimension reduction to parameter selection procedure.\n")
    # ----------------------- parameters for MachineLearning---------------------- #
    parse.add_argument('-ml', type=str, choices=Machine_Learning_Algorithm, required=False,
                       help="The machine learning algorithm, for example: Support Vector Machine(SVM).")
    parse.add_argument('-grid', type=int, nargs='*', choices=[0, 1], default=0,
                       help="grid = 0 for rough grid search, grid = 1 for meticulous grid search.")
    # parameters for svm
    parse.add_argument('-cost', type=int, nargs='*', help="Regularization parameter of 'SVM'.")
    parse.add_argument('-gamma', type=int, nargs='*', help="Kernel coefficient for 'rbf' of 'SVM'.")
    # parameters for rf
    parse.add_argument('-tree', type=int, nargs='*', help="The number of trees in the forest for 'RF'.")
    # ----------------------- parameters for DeepLearning---------------------- #
    parse.add_argument('-lr', type=float, default=0.01, help="The value of learning rate for deep learning.")
    parse.add_argument('-epochs', type=int, help="The epoch number for train deep model.")
    parse.add_argument('-batch_size', type=int, default=50, help="The size of mini-batch for deep learning.")
    parse.add_argument('-dropout', type=float, default=0.6, help="The value of dropout prob for deep learning.")
    # parameters for LSTM, GRU
    parse.add_argument('-hidden_dim', type=int, default=256,
                       help="The size of the intermediate (a.k.a., feed forward) layer.")
    parse.add_argument('-n_layer', type=int, default=2, help="The number of units for 'LSTM' and 'GRU'.")
    # parameters for CNN
    parse.add_argument('-out_channels', type=int, default=256, help="The number of output channels for 'CNN'.")
    parse.add_argument('-kernel_size', type=int, default=5, help="The size of stride for 'CNN'.")
    # parameters for Transformer and Weighted-Transformer
    parse.add_argument('-d_model', type=int, default=256,
                       help="The dimension of multi-head attention layer for Transformer or Weighted-Transformer.")
    parse.add_argument('-d_ff', type=int, default=1024,
                       help="The dimension of fully connected layer of Transformer or Weighted-Transformer.")
    parse.add_argument('-n_heads', type=int, default=4,
                       help="The number of heads for Transformer or Weighted-Transformer.")
    # parameters for Reformer
    parse.add_argument('-n_chunk', type=int, default=8,
                       help="The number of chunks for processing lsh attention.")
    parse.add_argument('-rounds', type=int, default=1024,
                       help="The number of rounds for multiple rounds of hashing to reduce probability that similar "
                            "items fall in different buckets.")
    parse.add_argument('-bucket_length', type=int, default=64,
                       help="Average size of qk per bucket, 64 was recommended in paper")
    # parameters for ML parameter selection and cross validation
    parse.add_argument('-metric', type=str, choices=['Acc', 'MCC', 'AUC', 'BAcc', 'F1'], default='Acc',
                       help="The metric for parameter selection")
    parse.add_argument('-cv', choices=['5', '10', 'j'], default='5',
                       help="The cross validation mode.\n"
                            "5 or 10: 5-fold or 10-fold cross validation.\n"
                            "j: (character 'j') jackknife cross validation.")
    parse.add_argument('-sp', type=str, choices=['none', 'over', 'under', 'combine'], default='none',
                       help="Select technique for oversampling.")
    # ----------------------- parameters for input and output ---------------------- #
    parse.add_argument('-seq_file', required=True, help="The input file in FASTA format.")
    parse.add_argument('-label_file', required=True, help="The corresponding label file.")
    parse.add_argument('-ind_seq_file', help="The independent test dataset in FASTA format.")
    parse.add_argument('-ind_label_file', help="The corresponding label file of independent test dataset.")
    parse.add_argument('-fixed_len', type=int,
                       help="The length of sequence will be fixed via cutting or padding. If you don't set "
                            "value for 'fixed_len', it will be the maximum length of all input sequences. ")
    # parameters for output
    parse.add_argument('-format', default='csv', choices=['tab', 'svm', 'csv', 'tsv'],
                       help="The output format (default = csv).\n"
                            "tab -- Simple format, delimited by TAB.\n"
                            "svm -- The libSVM training data format.\n"
                            "csv, tsv -- The format that can be loaded into a spreadsheet program.")
    parse.add_argument('-bp', type=int, choices=[0, 1], default=0,
                       help="Select use batch mode or not, the parameter will change the directory for generating file "
                            "based on the method you choose.")

    # parameters for mll methods
    mll_arg_parser(parse)

    argv = parse.parse_args()
    main(argv)
