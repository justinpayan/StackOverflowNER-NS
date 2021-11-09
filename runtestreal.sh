#!/usr/bin/env bash

LAM=$1
LR=$2
NUM_EPS=$3
ORDER=$4
LABEL_MAP_FILE_NAME=$5
MODEL_DIR_ROOT=$6
DATA_DIR=$7

python test.py --data_dir $DATA_DIR \
    --model_dir_root $MODEL_DIR_ROOT --seq_train_type lll \
    --tasks $ORDER --n_train_epochs $NUM_EPS --top_k_lm 20 --top_k_ner 20 \
    --lm_lambda $LAM --gen_lm_sample_percentage 0.2 \
    --label_map_file $DATA_DIR/${LABEL_MAP_FILE_NAME} --logging_steps 100 \
    --learning_rate $LR --use_crf --fp32 --real_sample --test_last_only