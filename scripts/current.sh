#!/bin/bash

data=(~/blm-mll/data/sequences/snoRNA.fasta ~/blm-mll/data/sequences/snoRNA_label.csv)
# test run
# mll_param_select_template
./scripts/test.sh BR RF br_rf ${data[*]} 8
./scripts/test.sh BR SVM br_svm ${data[*]} 8
./scripts/test.sh LP RF lp_rf ${data[*]} 8
./scripts/test.sh LP SVM lp_svm ${data[*]} 8
#./scripts/test.sh RAkELo RF rakelo_rf ${data[*]} 8
#./scripts/test.sh RAkELo SVM rakelo_svm ${data[*]} 8
#./scripts/test.sh RAkELd RF rakeld_rf ${data[*]} 8
#./scripts/test.sh RAkELd SVM rakeld_svm ${data[*]} 8
#./scripts/test.sh MLARAM mlaram ${data[*]} 8
#./scripts/test.sh MLkNN mlknn ${data[*]} 8
