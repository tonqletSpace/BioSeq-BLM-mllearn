#!/bin/bash
mll=$1
ml=$2
out_res=$3

# batch run starting
cd code/

# data
cpu=5
data=(~/blm-mll/data/polypeptide/train/seqs.fasta ~/blm-mll/data/polypeptide/val/seqs.fasta)
label=(~/blm-mll/data/polypeptide/train/labels.csv ~/blm-mll/data/polypeptide/val/labels.csv)
ind_data=~/blm-mll/data/polypeptide/test/seqs.fasta
ind_label=~/blm-mll/data/polypeptide/test/labels.csv

# RF
p_tree=(450)
# SVM
p_cost=(-5)
p_gamma=(0)
#RAkELo
p_mll_ls=(5 11 5)
p_mll_mc=(15)
# MLkNN, BRkNNaClassifier, BRkNNbClassifier
p_mll_k=(450)
p_mll_s=(0.1 1.0 0.4)
 MLARAM
p_mll_v=(0.0 1.0 0.3)
p_mll_t=(0.01 1.0 3)

default_cmd=(-cv 5 -category Protein -cpu ${cpu} -bp 1 -metric Acc\
 -seq_file ${data[*]} -label ${label[*]} -ind_seq_file ${ind_data} -ind_label_file ${ind_label})

function run_ml_methods() {
#  echo "python BioSeq-BLM_Seq_mllearn.py $* -mll ${mll} -ml ${ml} ${default_cmd[*]}\
#       -tree ${p_tree[*]}"; exit 0
  if [[ ${mll} = "BR" ||  ${mll} = "LP" ]]; then
    if [ ${ml} = "RF" ]; then
      python BioSeq-BLM_Seq_mllearn.py $* -mll ${mll} -ml ${ml} ${default_cmd[*]}\
       -tree ${p_tree[*]}
    elif [ ${ml} = "SVM" ]; then
      python BioSeq-BLM_Seq_mllearn.py $* -mll ${mll} -ml ${ml} ${default_cmd[*]}\
       -cost ${p_cost[*]} -gamma ${p_gamma[*]}
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

# modes=OHE or SR
#ohe_methods=(PSSM PSFM One-hot CS)
ohe_methods=(PSSM)
for md in ${ohe_methods[*]}; do
  blm_mode=(-mode OHE -method ${md} -fixed_len 4200)
  # fa=(-sn L1-normalize -dr PCA -np 512 -rdb dr)
  fa=(PCA -np 512 -rdb dr)
  run_ml_methods ${blm_mode[*]} ${fa[*]}
done

# generate params and evals in code/
# last param: is_ind
python extract_result_mll.py Seq/Protein "${out_res}" true

# place all results int results/out_res/
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
