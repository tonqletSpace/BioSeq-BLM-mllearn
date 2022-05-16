import multiprocessing
import os
import time

from CheckAll import Batch_Path_Seq, DeepLearning, Classification, Method_Semantic_Similarity, prepare4train_seq
from CheckAll import Method_One_Hot_Enc, Feature_Extract_Mode, check_contain_chinese, seq_sys_check, dl_params_check, \
    seq_feature_check, mode_params_check, results_dir_check, ml_params_check, make_params_dicts, Final_Path, All_Words
from FeatureAnalysis import fa_process
from FeatureExtractionMode.utils.utils_write import seq_file2one, gen_label_array, out_seq_file, out_ind_file, \
    opt_file_copy, out_dl_seq_file, create_all_seq_file, fixed_len_control
from FeatureExtractionSeq import one_seq_fe_process
from MachineLearningAlgorithm.Classification.dl_machine import dl_cv_process, dl_ind_process
from MachineLearningAlgorithm.utils.utils_read import files2vectors_seq, read_dl_vec4seq
from MachineLearningSeq import one_ml_process, params_select, ml_results, ind_ml_results


def mll_ml_fe_process(args):
    # 合并序列文件
    input_one_file = create_all_seq_file(args.seq_file, args.results_dir)
    # 统计样本数目和序列长度
    sp_num_list, seq_len_list = seq_file2one(args.category, args.seq_file, args.label, input_one_file)