#!/bin/bash
mll=$1
ml=$2
out_res=$3

# batch run starting
cd code/

# data
cpu=5
seq_files=~/blm-mll/data/bt_lncRNA/human_lncRNA.fasta
labels=~/blm-mll/data/bt_lncRNA/human_lncRNA_label.csv

# RF
p_tree=(350 800 200)
# SVM
p_cost=(-6 5 6)
p_gamma=(-4 3 4)
#RAkELo
p_mll_ls=(10)
p_mll_mc=(15)
# MLkNN, BRkNNaClassifier, BRkNNbClassifier
p_mll_k=(450)
p_mll_s=(0.1 1.0 0.4)
#  MLARAM
p_mll_v=(0.1 1.0 0.4)
p_mll_t=(0.1 1.0 0.4)

ml_default_cmd=(-cv 5 -category RNA -cpu ${cpu} -bp 1 -metric Acc -fixed_len 210
 -seq_file ${seq_files} -label ${labels}  -mix_mode as_rna)

# cnn best
dl_default_cmd=(-cv 5 -category RNA -cpu ${cpu} -bp 1 -metric Acc -fixed_len 210 -epochs 200 -batch_size 30 -lr 1e-4\
   -seq_file ${seq_files} -label ${labels} -mix_mode as_rna)

function run_ml_methods() {
  if [[ ${mll} = "BR" ||  ${mll} = "LP" ]]; then
    if [ ${ml} = "RF" ]; then
      python BioSeq-BLM_Seq_mllearn.py $* -mll ${mll} -ml ${ml} ${ml_default_cmd[*]}\
       -tree ${p_tree[*]}
    elif [ ${ml} = "SVM" ]; then
      python BioSeq-BLM_Seq_mllearn.py $* -mll ${mll} -ml ${ml} ${ml_default_cmd[*]}\
       -cost ${p_cost[*]} -gamma ${p_gamma[*]}
    # deep learning
    elif [ ${ml} = "CNN" ]; then
      python BioSeq-BLM_Seq_mllearn.py $* -mll ${mll} -ml ${ml} ${dl_default_cmd[*]}\
        -out_channels 1024 -kernel_size 4
    elif [[ ${ml} = "LSTM" || ${ml} = "GRU" ]]; then
      python BioSeq-BLM_Seq_mllearn.py $* -mll ${mll} -ml ${ml} ${dl_default_cmd[*]}\
        -hidden_dim 1024 -n_layer 2
    elif [[ ${ml} = "Transformer" || ${ml} = "Weighted-Transformer" ]]; then
      python BioSeq-BLM_Seq_mllearn.py $* -mll ${mll} -ml ${ml} ${dl_default_cmd[*]}\
        -n_layer 2 -d_model 256 -d_ff 1024 -n_heads 4
    else
      exit 1
    fi

  elif [ ${mll} = "RAkELo" ]; then
    if [ ${ml} = "RF" ]; then
      python BioSeq-BLM_Seq_mllearn.py $* -mll ${mll} -ml ${ml} ${ml_default_cmd[*]}\
       -mll_ls ${p_mll_ls[*]} -mll_mc ${p_mll_mc[*]} -tree ${p_tree[*]}
    elif [ ${ml} = "SVM" ]; then
      python BioSeq-BLM_Seq_mllearn.py $* -mll ${mll} -ml ${ml} ${ml_default_cmd[*]}\
       -mll_ls ${p_mll_ls[*]} -mll_mc ${p_mll_mc[*]} -cost ${p_cost[*]} -gamma ${p_gamma[*]}
    # deep learning
    elif [ ${ml} = "CNN" ]; then
      python BioSeq-BLM_Seq_mllearn.py $* -mll ${mll} -ml ${ml} ${dl_default_cmd[*]}\
        -mll_ls ${p_mll_ls[*]} -mll_mc ${p_mll_mc[*]} -out_channels 1024 -kernel_size 4
    elif [[ ${ml} = "LSTM" || ${ml} = "GRU" ]]; then
      python BioSeq-BLM_Seq_mllearn.py $* -mll ${mll} -ml ${ml} ${dl_default_cmd[*]}\
        -mll_ls ${p_mll_ls[*]} -mll_mc ${p_mll_mc[*]} -hidden_dim 1024 -n_layer 2
    elif [[ ${ml} = "Transformer" || ${ml} = "Weighted-Transformer" ]]; then
      python BioSeq-BLM_Seq_mllearn.py $* -mll ${mll} -ml ${ml} ${dl_default_cmd[*]}\
        -mll_ls ${p_mll_ls[*]} -mll_mc ${p_mll_mc[*]} -n_layer 2 -d_model 256 -d_ff 1024 -n_heads 4
    else
      exit 1
    fi

  elif [ ${mll} = "RAkELd" ]; then
    if [ ${ml} = "RF" ]; then
      python BioSeq-BLM_Seq_mllearn.py $* -mll ${mll} -ml ${ml} ${ml_default_cmd[*]}\
       -mll_ls ${p_mll_ls[*]} -tree ${p_tree[*]}
    elif [ ${ml} = "SVM" ]; then
      python BioSeq-BLM_Seq_mllearn.py $* -mll ${mll} -ml ${ml} ${ml_default_cmd[*]}\
       -mll_ls ${p_mll_ls[*]} -cost ${p_cost[*]} -gamma ${p_gamma[*]}
    # deep learning
    elif [ ${ml} = "CNN" ]; then
      python BioSeq-BLM_Seq_mllearn.py $* -mll ${mll} -ml ${ml} ${dl_default_cmd[*]}\
       -mll_ls ${p_mll_ls[*]} -out_channels 1024 -kernel_size 4
    elif [[ ${ml} = "LSTM" || ${ml} = "GRU" ]]; then
      python BioSeq-BLM_Seq_mllearn.py $* -mll ${mll} -ml ${ml} ${dl_default_cmd[*]}\
       -mll_ls ${p_mll_ls[*]} -hidden_dim 1024 -n_layer 2
    elif [[ ${ml} = "Transformer" || ${ml} = "Weighted-Transformer" ]]; then
      python BioSeq-BLM_Seq_mllearn.py $* -mll ${mll} -ml ${ml} ${dl_default_cmd[*]}\
        -mll_ls ${p_mll_ls[*]} -n_layer 2 -d_model 256 -d_ff 1024 -n_heads 4
    else
      exit 1
    fi

  else
    exit 1
  fi

  return 0
}

if [[ ${ml} = "CNN" || ${ml} = "LSTM" || ${ml} = "GRU" || ${ml} = "Transformer" || ${ml} = "Weighted-Transformer" ]]; then
    ohe_methods=(One-hot)
    for md in ${ohe_methods[*]}; do
      blm_mode=(-mode OHE -method ${md})
      run_ml_methods ${blm_mode[*]}
    done
else
    # BSLM based on BOW, TF-IDF, TextRank
    # 12 total
    bslm_modes=(BOW TF-IDF) # Attention TR
    dna_words=(Kmer)
    for md in ${bslm_modes[*]}; do
      for wd in ${dna_words[*]}; do
        blm_mode=(-mode ${md} -words ${wd})
        run_ml_methods ${blm_mode[*]}
      done
    done

    # tb6 BSLMs based on topic models
    # 9 total
    tm_methods=(LSA LDA)
    sub_methods=(BOW) # Attention TextRank
    for md in ${tm_methods[*]}; do
      for sub_md in ${sub_methods[*]}; do
        blm_mode=(-mode TM -method ${md} -in_tm ${sub_md} -words Kmer)
        run_ml_methods ${blm_mode[*]}
      done
    done

    # tb7 BNLMs based on word embedding
    # 4 total
    we_methods=(word2vec fastText)
    we_rna_words=(Kmer)
    for we in ${we_methods[*]}; do
      for wd in ${we_rna_words[*]}; do
        blm_mode=(-mode WE -method ${we} -words ${wd})
        run_ml_methods ${blm_mode[*]}
      done
    done
fi





# generate params and evals in code/
# last param: is_ind
# only useful for ml methods
# python extract_result_mll.py Seq/Protein "${out_res}" true

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
