LAM=$1
GEN_PER=$2
LR=$3
NUM_EPS=$4
ORDER=$5
LABEL_MAP_FILE_NAME=$6
SEQ_TRAIN_TYPE=$7
MODEL_DIR_ROOT=$8
DATA_DIR=$9
MODEL_BASE_DIR=${10}

python train.py --data_dir $DATA_DIR \
    --model_dir_root $MODEL_DIR_ROOT --seq_train_type $SEQ_TRAIN_TYPE \
    --tasks $ORDER --n_train_epochs $NUM_EPS --top_k_lm 20 --top_k_ner 20 \
    --lm_lambda $LAM --gen_lm_sample_percentage $GEN_PER \
    --label_map_file $DATA_DIR/${LABEL_MAP_FILE_NAME} --logging_steps 100 \
    --learning_rate $LR --use_crf --fp32 --add_task_tokens --use_task_in_ner \
    --model_base_dir $MODEL_BASE_DIR
#    --learning_rate $LR --use_crf --fp32 --add_task_tokens --use_task_in_ner --ic --min_n_steps 10000 \


#    --learning_rate $LR --use_crf --fp32 --add_task_tokens --use_task_in_ner --short_exs_debug &

# ./setupandrunexp.sh 0.25 0.2 6.25e-5 0 "wnut conll_eng" wnut_conll_labels finetune

# ./setupandrunexp.sh 0.25 0.2 6.25e-5 0 "so_1 so_2 so_3 so_4 so_5" so_episodes_dataset/so_labels lll so
# ./setupandrunexp.sh 0.25 0.2 6.25e-5 0 "so_1 so_2 so_3 so_4 so_5" so_episodes_dataset/so_labels finetune

# ./setupandrunexp.sh 0.25 0.2 6.25e-5 0 "so_1 so_2 so_3 so_4 so_5" so_data/so_labels lll /iesl/canvas/jpayan/Lamolrelease/models ~/Lamolrelease /iesl/canvas/jpayan/Lamolrelease