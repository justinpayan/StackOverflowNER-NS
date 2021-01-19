LAM=$1
GEN_PER=$2
LR=$3
NUM_EPS=$4
ORDER=$5
LABEL_MAP_FILE_NAME=$6
SEQ_TRAIN_TYPE=$7
MODEL_DIR_ROOT=$8
DATA_DIR=$9

#python test.py --data_dir /efs-storage/data \
#    --model_dir_root /efs-storage/models --seq_train_type lll \
#    --tasks $ORDER --n_train_epochs $NUM_EPS --top_k_lm 20 --top_k_ner 20 \
#    --lm_lambda $LAM --gen_lm_sample_percentage 0.2 \
#    --label_map_file /efs-storage/data/${LABEL_MAP_FILE_NAME} --logging_steps 100 \
#    --learning_rate $LR --use_crf --fp32 --test_last_only &

python test.py --data_dir $DATA_DIR \
    --model_dir_root $MODEL_DIR_ROOT --seq_train_type $SEQ_TRAIN_TYPE \
    --tasks $ORDER --n_train_epochs $NUM_EPS --top_k_lm 20 --top_k_ner 20 \
    --lm_lambda $LAM --gen_lm_sample_percentage $GEN_PER \
    --label_map_file $DATA_DIR/${LABEL_MAP_FILE_NAME} --logging_steps 100 \
    --learning_rate $LR --use_crf --fp32 --add_task_tokens --use_task_in_ner --test_last_only &
#    --learning_rate $LR --use_crf --fp32 --add_task_tokens --use_task_in_ner --test_last_only --ic &

# ./runtest.sh 0.25 0.2 6.25e-5 0 "wnut conll_eng" wnut_conll_labels lll
# ./runtest.sh 0.25 0.2 6.25e-5 0 "so_1 so_2 so_3 so_4 so_5" so_episodes_dataset/so_labels lll
# ./runtest.sh 0.25 0.2 6.25e-5 0 "so_1 so_2 so_3 so_4 so_5" so_episodes_dataset/so_labels finetune

# ./runtest.sh 0.25 0.2 6.25e-5 0 "so_1 so_2 so_3 so_4 so_5" so_data/so_labels lll /iesl/canvas/jpayan/Lamolrelease/models ~/Lamolrelease
# ./runtest.sh 0.25 0.2 6.25e-5 0 "so_1 so_2 so_3 so_4 so_5" so_data/so_labels finetune /iesl/canvas/jpayan/Lamolrelease/models ~/Lamolrelease