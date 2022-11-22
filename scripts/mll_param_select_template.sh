#!/bin/bash
mll=$1
ml=$2
seq_files=$3
labels=$4
cpu=$5

# intro
echo "into directory:"
pwd
echo "please run command below for mll tasks with BR(RF) algorithm:"
echo "./scripts/mll_param_select_template.sh BR RF ~/blm-mll/data/sequences/snoRNA.fasta ~/blm-mll/data/sequences/snoRNA_label.csv 8 test"

# batch run starting

cd code/
# RF
p_tree=(50 601 50)


# tb345 BSLM based on BOW, TF-IDF, TextRank
# 12 total
bslm_modes=(BOW TF-IDF TR) # Attention TR
dna_words=(Kmer RevKmer Mismatch Subsequence)
for md in ${bslm_modes[*]}; do
  for wd in ${dna_words[*]}; do
    python BioSeq-BLM_Seq_mllearn.py -category DNA -mode ${md} -words ${wd} -mll ${mll} -ml ${ml} -tree ${p_tree[*]} -seq_file ${seq_files} -label ${labels} -cpu ${cpu} -bp 1 -metric Acc
  done
done


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
python extract_result_mll.py Seq/DNA

# place all results int results/out_res/
out_res=$6
cd ../results/

if [ $# -eq 6 ]; then
  if [ -d "${out_res}" ]; then
    rm -rf "${out_res}"
  fi

  mv batch "${out_res}"

  echo "Done! find results in ${out_res}"
fi


