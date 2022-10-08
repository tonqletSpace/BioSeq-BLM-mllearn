#!/bin/bash

cd ../code

# Binary
# (BR) (SVM, RF)(CNN, GRU, trm)
python BioSeq-BLM_Res_mllearn.py -category Protein -method BLOSUM62 -mll BR -ml SVM -cost 2 -gamma 3 -window 3 -seq_file ../data/dev/mll_protein_seq.txt -label_file ../data/dev/mll_protein_label.txt -metric F1 -ind_seq_file ../data/dev/mll_protein_seq.txt -ind_label_file ../data/dev/mll_protein_label.txt

python BioSeq-BLM_Res_mllearn.py -category Protein -method BLOSUM62 -mll BR -ml RF -window 3 -seq_file ../data/dev/mll_protein_seq.txt -label_file ../data/dev/mll_protein_label.txt -metric F1 -ind_seq_file ../data/dev/mll_protein_seq.txt -ind_label_file ../data/dev/mll_protein_label.txt

python BioSeq-BLM_Res_mllearn.py -category Protein -method One-hot -mll BR -ml CNN -epochs 1 -out_channels 50 -kernel_size 3 -window 3 -seq_file ../data/dev/mll_protein_seq.txt -label_file ../data/dev/mll_protein_label.txt -metric F1 -ind_seq_file ../data/dev/mll_protein_seq.txt -ind_label_file ../data/dev/mll_protein_label.txt

python BioSeq-BLM_Res_mllearn.py -category Protein -method One-hot -mll BR -ml GRU -epochs 2 -hidden_dim 50 -n_layer 3 -window 3 -seq_file ../data/dev/mll_protein_seq.txt -label_file ../data/dev/mll_protein_label.txt -metric F1 -ind_seq_file ../data/dev/mll_protein_seq.txt -ind_label_file ../data/dev/mll_protein_label.txt

python BioSeq-BLM_Res_mllearn.py -category Protein -method One-hot -mll BR -ml  Transformer -epochs 2 -n_layer 2 -d_model 50 -d_ff 60 -n_heads 2 -window 3 -seq_file ../data/dev/mll_protein_seq.txt -label_file ../data/dev/mll_protein_label.txt -metric F1 -ind_seq_file ../data/dev/mll_protein_seq.txt -ind_label_file ../data/dev/mll_protein_label.txt

# (CC) (ml SVM, RF)
python BioSeq-BLM_Res_mllearn.py -category Protein -method BLOSUM62 -mll CC -ml SVM -window 3 -seq_file ../data/dev/mll_protein_seq.txt -label_file ../data/dev/mll_protein_label.txt -metric F1 -ind_seq_file ../data/dev/mll_protein_seq.txt -ind_label_file ../data/dev/mll_protein_label.txt

python BioSeq-BLM_Res_mllearn.py -category Protein -method BLOSUM62 -mll CC -ml RF -window 3 -seq_file ../data/dev/mll_protein_seq.txt -label_file ../data/dev/mll_protein_label.txt -metric F1 -ind_seq_file ../data/dev/mll_protein_seq.txt -ind_label_file ../data/dev/mll_protein_label.txt


# (LP) - (ml SVM, RF)(dl CNN, GRU, trm)
python BioSeq-BLM_Res_mllearn.py -category Protein -method BLOSUM62 -mll LP -ml SVM -window 3 -seq_file ../data/dev/mll_protein_seq.txt -label_file ../data/dev/mll_protein_label.txt -metric F1 -ind_seq_file ../data/dev/mll_protein_seq.txt -ind_label_file ../data/dev/mll_protein_label.txt

python BioSeq-BLM_Res_mllearn.py -category Protein -method BLOSUM62 -mll LP -ml RF -window 3 -seq_file ../data/dev/mll_protein_seq.txt -label_file ../data/dev/mll_protein_label.txt -metric F1 -ind_seq_file ../data/dev/mll_protein_seq.txt -ind_label_file ../data/dev/mll_protein_label.txt

python BioSeq-BLM_Res_mllearn.py -category Protein -method One-hot -mll LP -ml CNN -epochs 1 -out_channels 50 -kernel_size 3 -window 3 -seq_file ../data/dev/mll_protein_seq.txt -label_file ../data/dev/mll_protein_label.txt -metric F1 -ind_seq_file ../data/dev/mll_protein_seq.txt -ind_label_file ../data/dev/mll_protein_label.txt

python BioSeq-BLM_Res_mllearn.py -category Protein -method One-hot -mll LP -ml GRU -epochs 2 -hidden_dim 50 -n_layer 3 -window 3 -seq_file ../data/dev/mll_protein_seq.txt -label_file ../data/dev/mll_protein_label.txt -metric F1 -ind_seq_file ../data/dev/mll_protein_seq.txt -ind_label_file ../data/dev/mll_protein_label.txt

python BioSeq-BLM_Res_mllearn.py -category Protein -method One-hot -mll LP -ml Transformer -epochs 2 -n_layer 2 -d_model 50 -d_ff 60 -n_heads 2 -window 3 -seq_file ../data/dev/mll_protein_seq.txt -label_file ../data/dev/mll_protein_label.txt -metric F1 -ind_seq_file ../data/dev/mll_protein_seq.txt -ind_label_file ../data/dev/mll_protein_label.txt

python BioSeq-BLM_Res_mllearn.py -category Protein -method One-hot -mll LP -ml Weighted-Transformer -epochs 2 -n_layer 2 -d_model 50 -d_ff 60 -n_heads 2 -window 3 -seq_file ../data/dev/mll_protein_seq.txt -label_file ../data/dev/mll_protein_label.txt  -metric F1 -batch_size 30 -lr 1e-3 -fixed_len 500 -ind_seq_file ../data/dev/mll_protein_seq.txt -ind_label_file ../data/dev/mll_protein_label.txt

# MEKA(FW, RT, CLR) - (RF, SVM)
python BioSeq-BLM_Res_mllearn.py -category Protein -method BLOSUM62 -mll FW -ml RF -tree 5 -window 3 -seq_file ../data/dev/mll_protein_seq.txt -label_file ../data/dev/mll_protein_label.txt -metric F1 -ind_seq_file ../data/dev/mll_protein_seq.txt -ind_label_file ../data/dev/mll_protein_label.txt

python BioSeq-BLM_Res_mllearn.py -category Protein -method BLOSUM62 -mll FW -ml SVM -cost 1 -gamma 2 -window 3 -seq_file ../data/dev/mll_protein_seq.txt -label_file ../data/dev/mll_protein_label.txt -metric F1 -ind_seq_file ../data/dev/mll_protein_seq.txt -ind_label_file ../data/dev/mll_protein_label.txt

python BioSeq-BLM_Res_mllearn.py -category Protein -method BLOSUM62 -mll RT -ml RF -tree 5 -window 3 -seq_file ../data/dev/mll_protein_seq.txt -label_file ../data/dev/mll_protein_label.txt -metric F1 -ind_seq_file ../data/dev/mll_protein_seq.txt -ind_label_file ../data/dev/mll_protein_label.txt

python BioSeq-BLM_Res_mllearn.py -category Protein -method BLOSUM62 -mll RT -ml SVM -cost 1 -gamma 2 -window 3 -seq_file ../data/dev/mll_protein_seq.txt -label_file ../data/dev/mll_protein_label.txt -metric F1 -ind_seq_file ../data/dev/mll_protein_seq.txt -ind_label_file ../data/dev/mll_protein_label.txt

python BioSeq-BLM_Res_mllearn.py -category Protein -method BLOSUM62 -mll CLR -ml RF -tree 5 -window 3 -seq_file ../data/dev/mll_protein_seq.txt -label_file ../data/dev/mll_protein_label.txt -metric F1 -ind_seq_file ../data/dev/mll_protein_seq.txt -ind_label_file ../data/dev/mll_protein_label.txt

python BioSeq-BLM_Res_mllearn.py -category Protein -method BLOSUM62 -mll CLR -ml SVM -cost 1 -gamma 2 -window 3 -seq_file ../data/dev/mll_protein_seq.txt -label_file ../data/dev/mll_protein_label.txt -metric F1 -ind_seq_file ../data/dev/mll_protein_seq.txt -ind_label_file ../data/dev/mll_protein_label.txt

# Ensemble
# RAkELo, RF
python BioSeq-BLM_Res_mllearn.py -category Protein -method BLOSUM62 -mll RAkELo -ml RF --RAkEL_labelset_size 2 --RAkELo_model_count 3 -window 3 -seq_file ../data/dev/mll_protein_seq.txt -label_file ../data/dev/mll_protein_label.txt -metric F1 -ind_seq_file ../data/dev/mll_protein_seq.txt -ind_label_file ../data/dev/mll_protein_label.txt

# RAkELo, SVM
python BioSeq-BLM_Res_mllearn.py -category Protein -method BLOSUM62 -mll RAkELo -ml SVM -cost 2 -gamma 3 --RAkEL_labelset_size 2 --RAkELo_model_count 3 -window 3 -seq_file ../data/dev/mll_protein_seq.txt -label_file ../data/dev/mll_protein_label.txt -metric F1 -ind_seq_file ../data/dev/mll_protein_seq.txt -ind_label_file ../data/dev/mll_protein_label.txt

# RAkELo, CNN
python BioSeq-BLM_Res_mllearn.py -category Protein -method BLOSUM62 -mll RAkELo --RAkEL_labelset_size 2 --RAkELo_model_count 3 -ml CNN -epochs 1 -out_channels 50 -kernel_size 3 -window 3 -seq_file ../data/dev/mll_protein_seq.txt -label_file ../data/dev/mll_protein_label.txt -metric F1 -batch_size 30 -lr 1e-3 -fixed_len 500 -ind_seq_file ../data/dev/mll_protein_seq.txt -ind_label_file ../data/dev/mll_protein_label.txt

# RAkELo, gru
python BioSeq-BLM_Res_mllearn.py -category Protein -method BLOSUM62 -mll RAkELo --RAkEL_labelset_size 2 --RAkELo_model_count 3 -ml GRU -epochs 2 -hidden_dim 50 -n_layer 3 -window 3 -seq_file ../data/dev/mll_protein_seq.txt -label_file ../data/dev/mll_protein_label.txt -metric F1 -batch_size 30 -lr 1e-3 -fixed_len 500 -ind_seq_file ../data/dev/mll_protein_seq.txt -ind_label_file ../data/dev/mll_protein_label.txt

# RAkELo, trm
python BioSeq-BLM_Res_mllearn.py -category Protein -method BLOSUM62 -mll RAkELo --RAkEL_labelset_size 2 --RAkELo_model_count 3 -ml Transformer -epochs 1 -n_layer 2 -d_model 50 -d_ff 60 -n_heads 2 -window 3 -seq_file ../data/dev/mll_protein_seq.txt -label_file ../data/dev/mll_protein_label.txt -metric F1 -batch_size 30 -lr 1e-3  -fixed_len 500 -ind_seq_file ../data/dev/mll_protein_seq.txt -ind_label_file ../data/dev/mll_protein_label.txt

# RAkELd, RF
python BioSeq-BLM_Res_mllearn.py -category Protein -method BLOSUM62 -mll RAkELd -ml RF --RAkEL_labelset_size 2 -window 3 -seq_file ../data/dev/mll_protein_seq.txt -label_file ../data/dev/mll_protein_label.txt -metric F1 -ind_seq_file ../data/dev/mll_protein_seq.txt -ind_label_file ../data/dev/mll_protein_label.txt

# RAkELd, SVM
python BioSeq-BLM_Res_mllearn.py -category Protein -method BLOSUM62 -mll RAkELd -ml SVM -cost 2 -gamma 3 --RAkEL_labelset_size 2 -window 3 -seq_file ../data/dev/mll_protein_seq.txt -label_file ../data/dev/mll_protein_label.txt -metric F1 -ind_seq_file ../data/dev/mll_protein_seq.txt -ind_label_file ../data/dev/mll_protein_label.txt

# RAkELd, CNN
python BioSeq-BLM_Res_mllearn.py -category Protein -method BLOSUM62 -mll RAkELd --RAkEL_labelset_size 2 -ml CNN -epochs 1 -out_channels 50 -kernel_size 3 -window 3 -seq_file ../data/dev/mll_protein_seq.txt -label_file ../data/dev/mll_protein_label.txt -metric F1 -batch_size 30 -lr 1e-3 -fixed_len 500 -ind_seq_file ../data/dev/mll_protein_seq.txt -ind_label_file ../data/dev/mll_protein_label.txt

# RAkELd, gru
python BioSeq-BLM_Res_mllearn.py -category Protein -method BLOSUM62 -mll RAkELd --RAkEL_labelset_size 2 -ml GRU -epochs 2 -hidden_dim 50 -n_layer 3 -window 3 -seq_file ../data/dev/mll_protein_seq.txt -label_file ../data/dev/mll_protein_label.txt -metric F1 -batch_size 30 -lr 1e-3 -fixed_len 500 -ind_seq_file ../data/dev/mll_protein_seq.txt -ind_label_file ../data/dev/mll_protein_label.txt

# RAkELd, trm
python BioSeq-BLM_Res_mllearn.py -category Protein -method BLOSUM62 -mll RAkELd --RAkEL_labelset_size 2 -ml Transformer -epochs 1 -n_layer 2 -d_model 50 -d_ff 60 -n_heads 2 -window 3 -seq_file ../data/dev/mll_protein_seq.txt -label_file ../data/dev/mll_protein_label.txt -metric F1 -batch_size 30 -lr 1e-3  -fixed_len 500 -ind_seq_file ../data/dev/mll_protein_seq.txt -ind_label_file ../data/dev/mll_protein_label.txt

# Adaptation
# MLkNN
python BioSeq-BLM_Res_mllearn.py -category Protein -method BLOSUM62 -mll MLkNN -mll_k 3 5 -mll_s 0.6 0.8 0.1 -mll_ifn 2 5 2 -window 3 -seq_file ../data/dev/mll_protein_seq.txt -label_file ../data/dev/mll_protein_label.txt -metric F1 -fixed_len 500 -ind_seq_file ../data/dev/mll_protein_seq.txt -ind_label_file ../data/dev/mll_protein_label.txt

# BRkNNaClassifier
python BioSeq-BLM_Res_mllearn.py -category Protein -method BLOSUM62 -mll BRkNNaClassifier -mll_k 2 -window 3 -seq_file ../data/dev/mll_protein_seq.txt -label_file ../data/dev/mll_protein_label.txt -metric F1 -fixed_len 500 -ind_seq_file ../data/dev/mll_protein_seq.txt -ind_label_file ../data/dev/mll_protein_label.txt

# BRkNNbClassifier
python BioSeq-BLM_Res_mllearn.py -category Protein -method BLOSUM62 -mll BRkNNbClassifier -mll_k 2 6 2 -window 3 -seq_file ../data/dev/mll_protein_seq.txt -label_file ../data/dev/mll_protein_label.txt -metric F1 -fixed_len 500 -ind_seq_file ../data/dev/mll_protein_seq.txt -ind_label_file ../data/dev/mll_protein_label.txt

# MLARAM
python BioSeq-BLM_Res_mllearn.py -category Protein -method BLOSUM62 -mll MLARAM -window 3 -seq_file ../data/dev/mll_protein_seq.txt -label_file ../data/dev/mll_protein_label.txt -metric F1 -fixed_len 500 -ind_seq_file ../data/dev/mll_protein_seq.txt -ind_label_file ../data/dev/mll_protein_label.txt