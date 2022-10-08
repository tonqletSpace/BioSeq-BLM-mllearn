#!/bin/bash

cd ../code

# BR, RF
python BioSeq-BLM_Seq_mllearn.py -category DNA -mode OHE -method One-hot -mll BR -ml RF -tree 3 -seq_file ../data/dev/Endoplasmic_reticulum.fasta -bp 1 -metric F1 -ind_seq_file ../data/dev/Endoplasmic_reticulum.fasta

# BR SVM
python BioSeq-BLM_Seq_mllearn.py -category DNA -mode OHE -method One-hot -mll BR -ml SVM -cost 2 -gamma 3 -seq_file ../data/dev/Endoplasmic_reticulum.fasta -bp 1 -metric F1 -ind_seq_file ../data/dev/Endoplasmic_reticulum.fasta

# BR, cnn
python BioSeq-BLM_Seq_mllearn.py -category DNA -mode OHE -method One-hot -mll BR -ml CNN -epochs 2 -out_channels 50 -kernel_size 3 -seq_file ../data/dev/Endoplasmic_reticulum.fasta -bp 1 -metric F1 -batch_size 30 -lr 1e-3 -fixed_len 500 -ind_seq_file ../data/dev/Endoplasmic_reticulum.fasta

# BR, gru
python BioSeq-BLM_Seq_mllearn.py -category DNA -mode OHE -method One-hot -mll BR -ml GRU -epochs 2 -hidden_dim 50 -n_layer 3 -seq_file ../data/dev/Endoplasmic_reticulum.fasta -bp 1 -metric F1 -batch_size 30 -lr 1e-3 -fixed_len 500 -ind_seq_file ../data/dev/Endoplasmic_reticulum.fasta

# BR, trm
python BioSeq-BLM_Seq_mllearn.py -category DNA -mode OHE -method One-hot -mll BR -ml Transformer -epochs 2 -n_layer 2 -d_model 50 -d_ff 60 -n_heads 2 -seq_file ../data/dev/Endoplasmic_reticulum.fasta -bp 1 -metric F1 -batch_size 30 -lr 1e-3  -fixed_len 500 -ind_seq_file ../data/dev/Endoplasmic_reticulum.fasta

# BR，Weighted-Transformer
python BioSeq-BLM_Seq_mllearn.py -category DNA -mode OHE -method One-hot -mll BR -ml Weighted-Transformer -epochs 1 -n_layer 2 -d_model 50 -d_ff 60 -n_heads 2 -seq_file ../data/dev/Endoplasmic_reticulum.fasta -bp 1 -metric F1 -batch_size 30 -lr 1e-3  -fixed_len 500 -ind_seq_file ../data/dev/Endoplasmic_reticulum.fasta

# CC, RF
python BioSeq-BLM_Seq_mllearn.py -category DNA -mode OHE -method One-hot -mll CC -ml RF -tree 2 -seq_file ../data/dev/Endoplasmic_reticulum.fasta -bp 1 -metric F1 -ind_seq_file ../data/dev/Endoplasmic_reticulum.fasta
# CC, SVM
python BioSeq-BLM_Seq_mllearn.py -category DNA -mode OHE -method One-hot -mll CC -ml SVM -gamma 2 -cost 1 -seq_file ../data/dev/Endoplasmic_reticulum.fasta -bp 1 -metric F1 -ind_seq_file ../data/dev/Endoplasmic_reticulum.fasta


# mll LP, ml RF
python BioSeq-BLM_Seq_mllearn.py -category DNA -mode OHE -method One-hot -mll LP -ml RF -tree 2 -seq_file ../data/dev/Endoplasmic_reticulum.fasta -bp 1 -metric F1 -ind_seq_file ../data/dev/Endoplasmic_reticulum.fasta

# mll LP, ml SVM
python BioSeq-BLM_Seq_mllearn.py -category DNA -mode OHE -method One-hot -mll LP -ml SVM -cost 2 -gamma 3 -seq_file ../data/dev/Endoplasmic_reticulum.fasta -bp 1 -metric F1 -ind_seq_file ../data/dev/Endoplasmic_reticulum.fasta

# mll LP ml CNN
python BioSeq-BLM_Seq_mllearn.py -category DNA -mode OHE -method One-hot -mll LP -ml CNN -epochs 1 -out_channels 50 -kernel_size 3 -seq_file ../data/dev/Endoplasmic_reticulum.fasta -bp 1 -metric F1 -batch_size 30 -lr 1e-3 -fixed_len 500 -ind_seq_file ../data/dev/Endoplasmic_reticulum.fasta

# mll LP, ml lstm
python BioSeq-BLM_Seq_mllearn.py -category DNA -mode OHE -method One-hot -mll LP -ml LSTM -epochs 2 -hidden_dim 50 -n_layer 3 -seq_file ../data/dev/Endoplasmic_reticulum.fasta -bp 1 -metric F1 -batch_size 30 -lr 1e-3 -fixed_len 500 -ind_seq_file ../data/dev/Endoplasmic_reticulum.fasta

# mll LP, ml gru
python BioSeq-BLM_Seq_mllearn.py -category DNA -mode OHE -method One-hot -mll LP -ml GRU -epochs 2 -hidden_dim 50 -n_layer 3 -seq_file ../data/dev/Endoplasmic_reticulum.fasta -bp 1 -metric F1 -batch_size 30 -lr 1e-3 -fixed_len 500 -ind_seq_file ../data/dev/Endoplasmic_reticulum.fasta

# mll LP, ml trm
python BioSeq-BLM_Seq_mllearn.py -category DNA -mode OHE -method One-hot -mll LP -ml Transformer -epochs 2 -n_layer 2 -d_model 50 -d_ff 60 -n_heads 2 -seq_file ../data/dev/Endoplasmic_reticulum.fasta -bp 1 -metric F1 -batch_size 30 -lr 1e-3 -fixed_len 500 -ind_seq_file ../data/dev/Endoplasmic_reticulum.fasta

# mll LP, ml wt-trm
python BioSeq-BLM_Seq_mllearn.py -category DNA -mode OHE -method One-hot -mll LP -ml Weighted-Transformer -epochs 2 -n_layer 2 -d_model 50 -d_ff 60 -n_heads 2 -seq_file ../data/dev/Endoplasmic_reticulum.fasta -bp 1 -metric F1 -batch_size 30 -lr 1e-3 -fixed_len 500 -ind_seq_file ../data/dev/Endoplasmic_reticulum.fasta

# MEKA(FW, RT, CLR) WEKA(RF, SVM)
python BioSeq-BLM_Seq_mllearn.py -category DNA -mode OHE -method One-hot -mll FW -ml RF -tree 5 -seq_file ../data/dev/Endoplasmic_reticulum.fasta -bp 1 -metric F1 -ind_seq_file ../data/dev/Endoplasmic_reticulum.fasta

python BioSeq-BLM_Seq_mllearn.py -category DNA -mode OHE -method One-hot -mll FW -ml SVM -cost 1 -gamma 2 -seq_file ../data/dev/Endoplasmic_reticulum.fasta -bp 1 -metric F1 -ind_seq_file ../data/dev/Endoplasmic_reticulum.fasta

python BioSeq-BLM_Seq_mllearn.py -category DNA -mode OHE -method One-hot -mll RT -ml RF -tree 5 -seq_file ../data/dev/Endoplasmic_reticulum.fasta -bp 1 -metric F1 -ind_seq_file ../data/dev/Endoplasmic_reticulum.fasta

python BioSeq-BLM_Seq_mllearn.py -category DNA -mode OHE -method One-hot -mll RT -ml SVM -cost 1 -gamma 2 -seq_file ../data/dev/Endoplasmic_reticulum.fasta -bp 1 -metric F1 -ind_seq_file ../data/dev/Endoplasmic_reticulum.fasta

python BioSeq-BLM_Seq_mllearn.py -category DNA -mode OHE -method One-hot -mll CLR -ml RF -tree 5 -seq_file ../data/dev/Endoplasmic_reticulum.fasta -bp 1 -metric F1 -ind_seq_file ../data/dev/Endoplasmic_reticulum.fasta

python BioSeq-BLM_Seq_mllearn.py -category DNA -mode OHE -method One-hot -mll CLR -ml SVM -cost 1 -gamma 2 -seq_file ../data/dev/Endoplasmic_reticulum.fasta -bp 1 -metric F1 -ind_seq_file ../data/dev/Endoplasmic_reticulum.fasta

# RAkELo, RF
python BioSeq-BLM_Seq_mllearn.py -category DNA -mode OHE -method One-hot -mll RAkELo -ml RF -tree 2 --RAkEL_labelset_size 3 --RAkELo_model_count 6 -seq_file ../data/dev/Endoplasmic_reticulum.fasta -bp 1 -metric F1 -ind_seq_file ../data/dev/Endoplasmic_reticulum.fasta

# RAkELo, SVM
python BioSeq-BLM_Seq_mllearn.py -category DNA -mode OHE -method One-hot -mll RAkELo -ml SVM -cost 2 -gamma 3 --RAkEL_labelset_size 3 --RAkELo_model_count 6 -seq_file ../data/dev/Endoplasmic_reticulum.fasta -bp 1 -metric F1 -ind_seq_file ../data/dev/Endoplasmic_reticulum.fasta

# RAkELo, CNN
python BioSeq-BLM_Seq_mllearn.py -category DNA -mode OHE -method One-hot -mll RAkELo --RAkEL_labelset_size 3 --RAkELo_model_count 6 -ml CNN -epochs 1 -out_channels 50 -kernel_size 3 -seq_file ../data/dev/Endoplasmic_reticulum.fasta -bp 1 -metric F1 -batch_size 30 -lr 1e-3 -fixed_len 500 -ind_seq_file ../data/dev/Endoplasmic_reticulum.fasta

# RAkELo, gru
python BioSeq-BLM_Seq_mllearn.py -category DNA -mode OHE -method One-hot -mll RAkELo --RAkEL_labelset_size 3 --RAkELo_model_count 6 -ml GRU -epochs 2 -hidden_dim 50 -n_layer 3 -seq_file ../data/dev/Endoplasmic_reticulum.fasta -bp 1 -metric F1 -batch_size 30 -lr 1e-3 -fixed_len 500 -ind_seq_file ../data/dev/Endoplasmic_reticulum.fasta

# RAkELo, trm
python BioSeq-BLM_Seq_mllearn.py -category DNA -mode OHE -method One-hot -mll RAkELo --RAkEL_labelset_size 3 --RAkELo_model_count 3 -ml Transformer -epochs 1 -n_layer 2 -d_model 50 -d_ff 60 -n_heads 2 -seq_file ../data/dev/Endoplasmic_reticulum.fasta -bp 1 -metric F1 -batch_size 30 -lr 1e-3  -fixed_len 500 -ind_seq_file ../data/dev/Endoplasmic_reticulum.fasta

# RAkELd, RF
python BioSeq-BLM_Seq_mllearn.py -category DNA -mode OHE -method One-hot -mll RAkELd -ml RF --RAkEL_labelset_size 3 -seq_file ../data/dev/Endoplasmic_reticulum.fasta -bp 1 -metric F1 -ind_seq_file ../data/dev/Endoplasmic_reticulum.fasta

# RAkELd, SVM
 python BioSeq-BLM_Seq_mllearn.py -category DNA -mode OHE -method One-hot -mll RAkELd -ml SVM -cost 3 -gamma 7 --RAkEL_labelset_size 4 -seq_file ../data/dev/Endoplasmic_reticulum.fasta -bp 1 -metric F1 -ind_seq_file ../data/dev/Endoplasmic_reticulum.fasta

# RAkELd, CNN
python BioSeq-BLM_Seq_mllearn.py -category DNA -mode OHE -method One-hot -mll RAkELd --RAkEL_labelset_size 3 -ml CNN -epochs 1 -out_channels 50 -kernel_size 3 -seq_file ../data/dev/Endoplasmic_reticulum.fasta -bp 1 -metric F1 -batch_size 30 -lr 1e-3 -fixed_len 500 -ind_seq_file ../data/dev/Endoplasmic_reticulum.fasta

# RAkELd, gru
python BioSeq-BLM_Seq_mllearn.py -category DNA -mode OHE -method One-hot -mll RAkELd --RAkEL_labelset_size 3 -ml GRU -epochs 2 -hidden_dim 50 -n_layer 3 -seq_file ../data/dev/Endoplasmic_reticulum.fasta -bp 1 -metric F1 -batch_size 30 -lr 1e-3 -fixed_len 500 -ind_seq_file ../data/dev/Endoplasmic_reticulum.fasta

# RAkELd, trm
python BioSeq-BLM_Seq_mllearn.py -category DNA -mode OHE -method One-hot -mll RAkELd --RAkEL_labelset_size 3 -ml Transformer -epochs 1 -n_layer 2 -d_model 50 -d_ff 60 -n_heads 2 -seq_file ../data/dev/Endoplasmic_reticulum.fasta -bp 1 -metric F1 -batch_size 30 -lr 1e-3  -fixed_len 500 -ind_seq_file ../data/dev/Endoplasmic_reticulum.fasta

# RAkELd, wt-trm
python BioSeq-BLM_Seq_mllearn.py -category DNA -mode OHE -method One-hot -mll RAkELd --RAkEL_labelset_size 3 -ml Weighted-Transformer -epochs 2 -n_layer 2 -d_model 50 -d_ff 60 -n_heads 2 -seq_file ../data/dev/Endoplasmic_reticulum.fasta -bp 1 -metric F1 -batch_size 30 -lr 1e-3  -fixed_len 500 -ind_seq_file ../data/dev/Endoplasmic_reticulum.fasta

# MLkNN
python BioSeq-BLM_Seq_mllearn.py -category DNA -mode OHE -method One-hot -mll MLkNN -mll_k 8 -mll_s 1.0 -seq_file ../data/dev/Endoplasmic_reticulum.fasta -bp 1 -metric F1 -fixed_len 500 -ind_seq_file ../data/dev/Endoplasmic_reticulum.fasta

# BRkNNaClassifier
python BioSeq-BLM_Seq_mllearn.py -category DNA -mode OHE -method One-hot -mll BRkNNaClassifier -mll_k 3 -seq_file ../data/dev/Endoplasmic_reticulum.fasta -bp 1 -metric F1 -fixed_len 500 -ind_seq_file ../data/dev/Endoplasmic_reticulum.fasta

# BRkNNbClassifier
python BioSeq-BLM_Seq_mllearn.py -category DNA -mode OHE -method One-hot -mll BRkNNbClassifier -mll_k 2 -seq_file ../data/dev/Endoplasmic_reticulum.fasta -bp 1 -metric F1 -fixed_len 500 -ind_seq_file ../data/dev/Endoplasmic_reticulum.fasta

# MLARAM
python BioSeq-BLM_Seq_mllearn.py -category DNA -mode OHE -method One-hot -mll MLARAM -seq_file ../data/dev/Endoplasmic_reticulum.fasta -bp 1 -metric F1 -fixed_len 500 --MLARAM_vigilance 0 1 0.5 -mll_t 0.02 0.03 -ind_seq_file ../data/dev/Endoplasmic_reticulum.fasta