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
p_mll_k=(50 601 100)
p_mll_s=(0.1 1.0 0.3)
# BRkNNaClassifier
# BRkNNbClassifier
# MLARAM
p_mll_v=(0 1 0.3)
p_mll_t=(0.01 0.03 0.01)


default_cmd=(-category DNA -seq_file ${seq_files} -label ${labels} -cpu ${cpu} -bp 1 -metric Acc)

# tb345 BSLM based on BOW, TF-IDF, TextRank
# 12 total
bslm_modes=(BOW) # Attention TR  TF-IDF TR  RevKmer Mismatch Subsequence
dna_words=(Kmer)
for md in ${bslm_modes[*]}; do
  for wd in ${dna_words[*]}; do
    mode_words=(-mode ${md} -words ${wd})
    if [[ ${mll} = "BR" ||  ${mll} = "LP" ]]; then
      if [ ${ml} = "RF" ]; then
        python BioSeq-BLM_Seq_mllearn.py ${mode_words[*]} -mll ${mll} -ml ${ml} ${default_cmd[*]}\
         -tree ${p_tree[*]}
      elif [ ${ml} = "SVM" ]; then
        python BioSeq-BLM_Seq_mllearn.py ${mode_words[*]} -mll ${mll} -ml ${ml} ${default_cmd[*]}\
         -cost ${p_cost[*]} -gamma ${p_gamma[*]}
      else
        exit 1
      fi

    elif [ ${mll} = "RAkELo" ]; then
      if [ ${ml} = "RF" ]; then
        echo "BioSeq-BLM_Seq_mllearn.py ${mode_words[*]} -mll ${mll} -ml ${ml} ${default_cmd[*]}\
         -mll_ls ${p_mll_ls[*]} -mll_mc ${p_mll_mc[*]} -tree ${p_tree[*]}"
        python BioSeq-BLM_Seq_mllearn.py ${mode_words[*]} -mll ${mll} -ml ${ml} ${default_cmd[*]}\
         -mll_ls ${p_mll_ls[*]} -mll_mc ${p_mll_mc[*]} -tree ${p_tree[*]}
      elif [ ${ml} = "SVM" ]; then
        python BioSeq-BLM_Seq_mllearn.py ${mode_words[*]} -mll ${mll} -ml ${ml} ${default_cmd[*]}\
         -mll_ls ${p_mll_ls[*]} -mll_mc ${p_mll_mc[*]} -cost ${p_cost[*]} -gamma ${p_gamma[*]}
      else
        exit 1
      fi

    elif [ ${mll} = "RAkELd" ]; then
      if [ ${ml} = "RF" ]; then
        python BioSeq-BLM_Seq_mllearn.py ${mode_words[*]} -mll ${mll} -ml ${ml} ${default_cmd[*]}\
         -mll_ls ${p_mll_ls[*]} -tree ${p_tree[*]}
      elif [ ${ml} = "SVM" ]; then
        python BioSeq-BLM_Seq_mllearn.py ${mode_words[*]} -mll ${mll} -ml ${ml} ${default_cmd[*]}\
         -mll_ls ${p_mll_ls[*]} -cost ${p_cost[*]} -gamma ${p_gamma[*]}
      else
        exit 1
      fi

    elif [ ${mll} = "MLkNN" ]; then
      python BioSeq-BLM_Seq_mllearn.py ${mode_words[*]} -mll ${mll} ${default_cmd[*]}\
       -mll_k ${p_mll_k[*]} -mll_s ${p_mll_s[*]}

    elif [ ${mll} = "MLARAM" ]; then
      python BioSeq-BLM_Seq_mllearn.py ${mode_words[*]} -mll ${mll} ${default_cmd[*]}\
        -mll_t ${p_mll_t[*]} -mll_v ${p_mll_v[*]}

    else
      exit 1
    fi
    exit 0  # test
  done
done
exit 0

# tb6 BSLMs based on topic models
# 9 total
tm_methods=(LSA LDA PLSA)
sub_methods=(BOW TF-IDF TextRank) # Attention TextRank
for md in ${tm_methods[*]}; do
  for sub_md in ${sub_methods[*]}; do
    python BioSeq-BLM_Seq_mllearn.py -category DNA -mode TM -method ${md} -in_tm ${sub_md} -words Kmer -mll ${mll} -ml ${ml} -tree ${p_tree[*]} -seq_file ${seq_files} -label ${labels} -cpu ${cpu} -bp 1 -metric Acc
  done
done

# tb7 BNLMs based on word embedding
# 12 total
we_methods=(word2vec GloVe fastText)
we_dna_words=(Kmer RevKmer)  # bugs for the other two words in BioSeq-BLM
for we in ${we_methods[*]}; do
  for wd in ${we_dna_words[*]}; do
    python BioSeq-BLM_Seq_mllearn.py -category DNA -mode WE -method ${we} -words ${wd} -mll ${mll} -ml ${ml} -tree ${p_tree[*]} -seq_file ${seq_files} -label ${labels} -cpu ${cpu} -bp 1 -metric Acc
  done
done

# generate params and evals in code/
python extract_result_mll.py Seq/DNA "${out_res}"

# place all results int results/out_res/
cd ../results/

if [ -d "${out_res}" ]; then
  rm -rf "${out_res}"
fi

mv batch "${out_res}"

echo "Done! find results in ${out_res}"
