#!/bin/bash
mll=$1
ml=$2
out_res=$3
seq_files=$4
labels=$5
cpu=$6


# intro
echo "into directory:"
pwd
echo "please run command below for mll tasks with BR(RF) algorithm:"
echo "./scripts/mll_param_select_template.sh BR RF test ~/blm-mll/data/sequences/snoRNA.fasta ~/blm-mll/data/sequences/snoRNA_label.csv 8"

# batch run starting

cd code/
# RF
p_tree=(50 601 50)

default_cmd=(-category DNA -seq_file ${seq_files} -label ${labels} -cpu ${cpu} -bp 1 -metric Acc)

# tb345 BSLM based on BOW, TF-IDF, TextRank
# 12 total
bslm_modes=(BOW) # Attention TR  TF-IDF TR  RevKmer Mismatch Subsequence
dna_words=(Kmer)
for md in ${bslm_modes[*]}; do
  for wd in ${dna_words[*]}; do
    if [ ${ml} -eq "RF" ]; then
      python BioSeq-BLM_Seq_mllearn.py -mode ${md} -words ${wd} -mll ${mll} -ml ${ml} -tree ${p_tree[*]} ${default_cmd[*]}
    elif [ ${ml} -eq "SVM" ]; then
      python BioSeq-BLM_Seq_mllearn.py -category DNA -mode ${md} -words ${wd} -mll ${mll} -ml ${ml} -tree ${p_tree[*]} -seq_file ${seq_files} -label ${labels} -cpu ${cpu} -bp 1 -metric Acc
    else
      exit 1
    fi

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
