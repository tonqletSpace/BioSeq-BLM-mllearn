#!/bin/bash


# 多肽识别
./scripts/run_polypeptide.sh BR RF br_rf
exit 0
./scripts/run_polypeptide.sh BR SVM br_svm
./scripts/run_polypeptide.sh LP RF lp_rf
./scripts/run_polypeptide.sh LP SVM lp_svm
./scripts/run_polypeptide.sh RAkELo RF rakelo_rf
./scripts/run_polypeptide.sh RAkELo SVM rakelo_svm
./scripts/run_polypeptide.sh RAkELd RF rakeld_rf
./scripts/run_polypeptide.sh RAkELd SVM rakeld_svm
#./scripts/run_polypeptide.sh MLARAM _ mlaram
./scripts/run_polypeptide.sh MLkNN _ mlknn
./scripts/run_polypeptide.sh BRkNNaClassifier _ brknna
./scripts/run_polypeptide.sh BRkNNbClassifier _ brknnb

# 亚细胞定位
#data=(~/blm-mll/data/sequences/snoRNA.fasta ~/blm-mll/data/sequences/snoRNA_label.csv)
#data=(~/blm-mll/data/sequences/miRNA.fasta ~/blm-mll/data/sequences/miRNA_label.csv)

# mll_param_select_template
#./scripts/test.sh BR RF br_rf ${data[*]} 5
#./scripts/test.sh BR SVM br_svm ${data[*]} 5
#./scripts/test.sh LP RF lp_rf ${data[*]} 5
#./scripts/test.sh LP SVM lp_svm ${data[*]} 5
#./scripts/test.sh RAkELo RF rakelo_rf ${data[*]} 5
#./scripts/test.sh RAkELo SVM rakelo_svm ${data[*]} 5
#./scripts/test.sh RAkELd RF rakeld_rf ${data[*]} 5
#./scripts/test.sh RAkELd SVM rakeld_svm ${data[*]} 5
#./scripts/test.sh MLARAM _ mlaram ${data[*]} 5
#./scripts/test.sh MLkNN _ mlknn ${data[*]} 5
#./scripts/test.sh BRkNNaClassifier _ brknna ${data[*]} 5
#./scripts/test.sh BRkNNbClassifier _ brknnb ${data[*]} 5
