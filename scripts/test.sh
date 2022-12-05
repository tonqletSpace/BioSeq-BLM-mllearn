#!/bin/bash
mll=$1
ml=$2
out_res=$3
seq_files=$4
labels=$5
cpu=$6

# batch run starting

cd code/

# 有些blocked了
# RF 2
p_tree=(450)
# SVM 2
p_cost=(-5 1 5)
p_gamma=(0)
#RAkELo 3
p_mll_ls=(2 5 1)
p_mll_mc=(15)
#RAkELd 4*3
# p_mll_ls
# MLkNN
# BRkNNaClassifier
# BRkNNbClassifier
p_mll_k=(450)
p_mll_s=(0.1 1.0 0.4)
# MLARAM
p_mll_v=(0.0 1.0 0.3)
p_mll_t=(0.01 1.0 3)

default_cmd=(-cv 10 -category DNA -seq_file ${seq_files} -label ${labels} -cpu ${cpu} -bp 1 -metric Acc -mix_mode as_dna)

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
# lncRNA very slow
# dna_words=(Kmer RevKmer Mismatch Subsequence)
dna_words=(Kmer)
for md in ${bslm_modes[*]}; do
  for wd in ${dna_words[*]}; do
    blm_mode=(-mode ${md} -words ${wd} -word_size 4)
    run_ml_methods ${blm_mode[*]}
  done
done

# tb6 BSLMs based on topic models
# 9 total
tm_methods=(LSA LDA PLSA)
sub_methods=(BOW TF-IDF TextRank) # Attention TextRank
for md in ${tm_methods[*]}; do
  for sub_md in ${sub_methods[*]}; do
    blm_mode=(-mode TM -method ${md} -in_tm ${sub_md} -words Kmer -word_size 4)
    run_ml_methods ${blm_mode[*]}
  done
done

# tb7 BNLMs based on word embedding
# 12 total
we_methods=(word2vec Glove fastText)
we_dna_words=(Kmer)  # bugs found for the other two words in BioSeq-BLM
for we in ${we_methods[*]}; do
  for wd in ${we_dna_words[*]}; do
    blm_mode=(-mode WE -method ${we} -words ${wd} -word_size 4)
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
