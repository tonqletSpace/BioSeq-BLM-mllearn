#!/bin/bash
mll=$1
ml=$2
out_res=$3

seq_files=~/blm-mll/data/bt_lncRNA/human_lncRNA.fasta
labels=~/blm-mll/data/bt_lncRNA/human_lncRNA_label.csv
cpu=5

# batch run starting

cd code/
# RF
# p_tree=(50 600 100)
p_tree=(50 100 50)  # debug!
# SVM
# p_cost=(-12 12 4)
# p_gamma=(-12 12 4)
p_cost=(-2 2 2)
p_gamma=(-2 2 2)  # debug!

#RAkELo
p_mll_ls=(2 5)
p_mll_mc=(3 20 4)
#RAkELd
# p_mll_ls
# MLkNN
# BRkNNaClassifier
# BRkNNbClassifier
p_mll_k=(30 165 40)
p_mll_s=(0.1 1.0 0.3)
# MLARAM
p_mll_v=(0 1 0.3)
p_mll_t=(0.01 0.03 0.01)

default_cmd=(-cv 10 -category RNA -seq_file ${seq_files} -label ${labels} -cpu ${cpu} -bp 1 -metric Acc -mix_mode as_rna)

function run_ml_methods() {
  if [[ ${mll} = "BR" ||  ${mll} = "LP" ||  ${mll} = "CC" ||  ${mll} = "RT" ||  ${mll} = "CLR" ||  ${mll} = "FW" ]]; then
    if [ ${ml} = "RF" ]; then
      python BioSeq-BLM_Seq_mllearn.py $* -mll ${mll} -ml ${ml} ${default_cmd[*]}\
       -tree ${p_tree[*]}
    elif [ ${ml} = "SVM" ]; then
      python BioSeq-BLM_Seq_mllearn.py $* -mll ${mll} -ml ${ml} ${default_cmd[*]}\
       -cost ${p_cost[*]} -gamma ${p_gamma[*]}
    elif [[ ${ml} = "NB" || ${ml} = "AB" || ${ml} = "kNN" || ${ml} = "BG" ]]; then
      python BioSeq-BLM_Seq_mllearn.py $* -mll ${mll} -ml ${ml} ${default_cmd[*]}
    else
      exit 1
    fi

  elif [ ${mll} = "RAkELo" ]; then
    if [ ${ml} = "RF" ]; then
      python BioSeq-BLM_Seq_mllearn.py $* -mll ${mll} -ml ${ml} ${default_cmd[*]}\
       -mll_ls ${p_mll_ls[*]} -mll_mc ${p_mll_mc[*]} -tree ${p_tree[*]}
    elif [ ${ml} = "SVM" ]; then
      python BioSeq-BLM_Seq_mllearn.py $* -mll ${mll} -ml ${ml} ${default_cmd[*]}\
       -mll_ls ${p_mll_ls[*]} -mll_mc ${p_mll_mc[*]} -cost ${p_cost[*]} -gamma ${p_gamma[*]}
    else
      exit 1
    fi

  elif [ ${mll} = "RAkELd" ]; then
    if [ ${ml} = "RF" ]; then
      python BioSeq-BLM_Seq_mllearn.py $* -mll ${mll} -ml ${ml} ${default_cmd[*]}\
       -mll_ls ${p_mll_ls[*]} -tree ${p_tree[*]}
    elif [ ${ml} = "SVM" ]; then
      python BioSeq-BLM_Seq_mllearn.py $* -mll ${mll} -ml ${ml} ${default_cmd[*]}\
       -mll_ls ${p_mll_ls[*]} -cost ${p_cost[*]} -gamma ${p_gamma[*]}
    else
      exit 1
    fi

  elif [ ${mll} = "MLARAM" ]; then
    python BioSeq-BLM_Seq_mllearn.py $* -mll ${mll} ${default_cmd[*]}\
     -mll_t ${p_mll_t[*]} -mll_v ${p_mll_v[*]}

  elif [ ${mll} = "MLkNN" ]; then
    python BioSeq-BLM_Seq_mllearn.py $* -mll ${mll} ${default_cmd[*]}\
     -mll_k ${p_mll_k[*]} -mll_s ${p_mll_s[*]}

  elif [[ ${mll} = "BRkNNaClassifier" || ${mll} = "BRkNNbClassifier" ]]; then
    python BioSeq-BLM_Seq_mllearn.py $* -mll ${mll} ${default_cmd[*]}\
     -mll_k ${p_mll_k[*]}

  else
    exit 1
  fi

  return 0
}

modes=(SR) # syntax rule
words=(DCC DAC) # DAC
for md in ${modes[*]}; do
  for wd in ${words[*]}; do
    blm_mode=(-mode ${md} -method ${wd})
    run_ml_methods ${blm_mode[*]}
  done
done

# run_ml_methods -mode ${md} -method ${wd} ${blm_mode[*]}


# generate params and evals in code/
python extract_result_mll.py Seq/RNA "${out_res}" Fasle

# place all results into results/out_res/
cd ../results/

# 汇总到tmp，由客户命名
if [ ! -d "tmp" ]; then
  mkdir "tmp"
fi

default_dir="tmp/${out_res}"
# 覆盖结果
if [ -d ${default_dir} ]; then
  rm -rf ${default_dir}
fi
mv batch ${default_dir}

echo "Done! results can be found in ${default_dir}"
