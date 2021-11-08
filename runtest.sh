#!/usr/bin/env bash

LAM=$1
GEN_PER=$2
LR=$3
NUM_EPS=$4
ORDER=$5
LABEL_MAP_FILE_NAME=$6
SEQ_TRAIN_TYPE=$7
MODEL_DIR_ROOT=$8
DATA_DIR=$9

python test.py --data_dir $DATA_DIR \
    --model_dir_root $MODEL_DIR_ROOT --seq_train_type $SEQ_TRAIN_TYPE \
    --tasks $ORDER --n_train_epochs $NUM_EPS --top_k_lm 20 --top_k_ner 20 \
    --lm_lambda $LAM --gen_lm_sample_percentage $GEN_PER \
    --label_map_file $DATA_DIR/${LABEL_MAP_FILE_NAME} --logging_steps 100 \
    --learning_rate $LR --use_crf --fp32 --test_last_only