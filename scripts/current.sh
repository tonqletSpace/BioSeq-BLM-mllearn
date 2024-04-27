#!/bin/bash
mode=$1

# ./scripts/current.sh aav
if [ ${mode} = 'ap' ]; then # analysis platform
    # 表征学习！！
    # blm
    ./scripts/mll_algo_arch_validation.sh BR ? br_?

    # platform的
    ./scripts/mll_algo_arch_validation.sh LP RF lp_rf
    ./scripts/mll_algo_arch_validation.sh LP SVM lp_svm
    ./scripts/mll_algo_arch_validation.sh LP CNN lp_rf
    ./scripts/mll_algo_arch_validation.sh RAkELo RF rakelo_rf
    ./scripts/mll_algo_arch_validation.sh RAkELo SVM rakelo_svm
    ./scripts/mll_algo_arch_validation.sh RAkELd RF rakeld_rf
    ./scripts/mll_algo_arch_validation.sh RAkELd SVM rakeld_svm

elif [ ${mode} = 'aav' ]; then # algo Arch validation

    ./scripts/mll_algo_arch_validation.sh CC RF cc_rf
    ./scripts/mll_algo_arch_validation.sh CC SVM cc_svm
    ./scripts/mll_algo_arch_validation.sh LP RF lp_rf
    ./scripts/mll_algo_arch_validation.sh LP SVM lp_svm
    ./scripts/mll_algo_arch_validation.sh RAkELo RF rakelo_rf
    ./scripts/mll_algo_arch_validation.sh RAkELo SVM rakelo_svm
    ./scripts/mll_algo_arch_validation.sh RAkELd RF rakeld_rf
    ./scripts/mll_algo_arch_validation.sh RAkELd SVM rakeld_svm
    ./scripts/mll_algo_arch_validation.sh CLR RF clr_rf
    ./scripts/mll_algo_arch_validation.sh CLR SVM clr_svm
    ./scripts/mll_algo_arch_validation.sh FW RF fw_rf
    ./scripts/mll_algo_arch_validation.sh FW SVM fw_svm
    ./scripts/mll_algo_arch_validation.sh RT RF rt_rf
    ./scripts/mll_algo_arch_validation.sh RT SVM rt_svm

    ./scripts/mll_algo_arch_validation.sh MLARAM _ mlaram
    ./scripts/mll_algo_arch_validation.sh MLkNN _ mlknn
    ./scripts/mll_algo_arch_validation.sh BRkNNaClassifier _ brknna
    ./scripts/mll_algo_arch_validation.sh BRkNNbClassifier _ brknnb

    ./scripts/mll_algo_arch_validation.sh BR RF br_rf
    ./scripts/mll_algo_arch_validation.sh BR SVM br_svm
    ./scripts/mll_algo_arch_validation.sh BR NB br_nb
    ./scripts/mll_algo_arch_validation.sh BR AB br_ab
    ./scripts/mll_algo_arch_validation.sh BR BG br_bg
    ./scripts/mll_algo_arch_validation.sh BR kNN br_knn

# ./scripts/current.sh db
elif [ ${mode} = 'db' ]; then # debug
    ./scripts/mll_algo_arch_validation.sh CLR RF clr_rf
    # ./scripts/mll_algo_arch_validation.sh CLR SVM clr_svm
    exit 0

elif [ ${mode} = 'sc' ]; then # subcell

    # 亚细胞定位
    #data=(~/blm-mll/data/sequences/snoRNA.fasta ~/blm-mll/data/sequences/snoRNA_label.csv)
    #data=(~/blm-mll/data/sequences/miRNA.fasta ~/blm-mll/data/sequences/miRNA_label.csv)

    # mll_param_select_template
    #./scripts/test.sh BR RF br_rf
    #./scripts/test.sh BR SVM br_svm
    # ./scripts/test.sh LP RF lp_rf_221226
    #./scripts/test.sh LP SVM lp_svm
    ./scripts/test.sh RAkELo RF rakelo_rf_221226
    #./scripts/test.sh RAkELo SVM rakelo_svm
    ./scripts/test.sh RAkELd RF rakeld_rf_221226
    #./scripts/test.sh RAkELd SVM rakeld_svm
    #./scripts/test.sh MLARAM _ mlaram
    #./scripts/test.sh MLkNN _ mlknn
    #./scripts/test.sh BRkNNaClassifier _ brknna
    #./scripts/test.sh BRkNNbClassifier _ brknnb
    exit 0

elif [ ${mode} = 'pp' ]; then # polypeptide


    # 多肽识别
    # ./scripts/run_polypeptide.sh LP RF lp_rf_221214
    # ./scripts/run_polypeptide.sh BR RF br_rf
    # ./scripts/run_polypeptide.sh BR SVM br_svm
    # ./scripts/run_polypeptide.sh LP SVM lp_svm_221214
    # ./scripts/run_polypeptide.sh RAkELo RF rakelo_rf_221214
    # ./scripts/run_polypeptide.sh RAkELo SVM rakelo_svm
    # ./scripts/run_polypeptide.sh RAkELd RF rakeld_rf
    # ./scripts/run_polypeptide.sh RAkELd SVM rakeld_svm
    #./scripts/run_polypeptide.sh MLARAM _ mlaram
    # ./scripts/run_polypeptide.sh MLkNN _ mlknn
    # ./scripts/run_polypeptide.sh BRkNNaClassifier _ brknna
    # ./scripts/run_polypeptide.sh BRkNNbClassifier _ brknnb

    ./scripts/run_polypeptide.sh LP CNN lp_cnn_221214
    # ./scripts/run_polypeptide.sh LP LSTM lp_lstm_221214
    # ./scripts/run_polypeptide.sh LP GRU lp_gru_221214
    # ./scripts/run_polypeptide.sh LP Transformer lp_trm_221214
    # ./scripts/run_polypeptide.sh LP Weighted-Transformer lp_wtrm_221214
    # exit 0

    # ./scripts/run_polypeptide.sh BR CNN
    # ./scripts/run_polypeptide.sh BR LSTM
    # ./scripts/run_polypeptide.sh BR GRU
    # ./scripts/run_polypeptide.sh BR Transformer
    # ./scripts/run_polypeptide.sh BR Weighted-Transformer

elif [ ${mode} = 'ex' ]; then # extraction
    target=(subcell/human_snoRNA_res subcell/lncRNA_res subcell/miRNA_res subcell/snoRNA_res)
    for dir in ${target[*]}; do
        # run scirpt in root directory(../scripts)
        cd code/
        python extract_result_mll.py ${dir} all false
    done
fi



