#!/bin/bash
mll=$1
ml=$2
out_res=$3
seq_files=$4
labels=$5
cpu=$6

# batch run starting

cd code/
# RF
p_tree=(50 601 100)
# SVM
p_cost=(-12 12 4)
p_gamma=(-12 12 4)
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

default_cmd=(-category DNA -seq_file ${seq_files} -label ${labels} -cpu ${cpu} -bp 1 -metric Acc)

function run_ml_methods() {
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


# tb345 BSLM based on BOW, TF-IDF, TextRank
# 12 total
bslm_modes=(BOW TF-IDF TR) # Attention TR
dna_words=(Kmer RevKmer Mismatch Subsequence)
for md in ${bslm_modes[*]}; do
  for wd in ${dna_words[*]}; do
    blm_mode=(-mode ${md} -words ${wd})
    run_ml_methods ${blm_mode[*]}
  done
done

# tb6 BSLMs based on topic models
# 9 total
tm_methods=(LSA LDA PLSA)
sub_methods=(BOW TF-IDF TextRank) # Attention TextRank
for md in ${tm_methods[*]}; do
  for sub_md in ${sub_methods[*]}; do
    blm_mode=(-mode TM -method ${md} -in_tm ${sub_md} -words Kmer)
    run_ml_methods ${blm_mode[*]}
  done
done

# tb7 BNLMs based on word embedding
# 12 total
we_methods=(word2vec GloVe fastText)
we_dna_words=(Kmer RevKmer)  # bugs found for the other two words in BioSeq-BLM
for we in ${we_methods[*]}; do
  for wd in ${we_dna_words[*]}; do
    blm_mode=(-mode WE -method ${we} -words ${wd})
    run_ml_methods ${blm_mode[*]}
  done
done

# generate params and evals in code/
python extract_result_mll.py Seq/DNA "${out_res}"

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
