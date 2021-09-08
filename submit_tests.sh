#!/usr/bin/env bash

TIME=$1

OUTBASE=/mnt/nfs/scratch1/jpayan/Lamolrelease

mkdir -p $OUTBASE/logs/${TIME}/test/lll
mkdir -p $OUTBASE/logs/${TIME}/test/real
mkdir -p $OUTBASE/logs/${TIME}/test/finetune

#sbatch -J test_real_so \
#        -e $OUTBASE/logs/${TIME}/test/real/lamol_test.err \
#        -o $OUTBASE/logs/${TIME}/test/real/lamol_test.log \
#        --mem=10G \
#        --partition=m40-short \
#        --time=01:00:00 \
#        --gres=gpu:1 \
#        ./runtestreal.sh 0.25 6.25e-5 0 "so_1 so_2 so_3 so_4 so_5" so_data/so_labels $OUTBASE/models/${TIME} ~/Lamolrelease
#
#sbatch -J test_ft_so \
#        -e $OUTBASE/logs/${TIME}/test/finetune/lamol_test.err \
#        -o $OUTBASE/logs/${TIME}/test/finetune/lamol_test.log \
#        --mem=15G \
#        --partition=m40-short \
#        --time=01:00:00 \
#        --gres=gpu:1 \
#        ./runtest.sh 0.25 0.2 6.25e-5 0 "so_1 so_2 so_3 so_4 so_5" so_data/so_labels finetune $OUTBASE/models/${TIME} ~/Lamolrelease
#
## Train on whole so train set, for getting a baseline
#sbatch -J test_base_so \
#        -e $OUTBASE/logs/${TIME}/test/finetune/lamol_test_whole_so.err \
#        -o $OUTBASE/logs/${TIME}/test/finetune/lamol_test_whole_so.log \
#        --mem=15G \
#        --partition=m40-short \
#        --time=01:00:00 \
#        --gres=gpu:1 \
#        ./runtest.sh 0.25 0.2 6.25e-5 0 "so_all_1 so_all_2 so_all_3 so_all_4 so_all_5" so_data/so_labels finetune $OUTBASE/models/${TIME} ~/Lamolrelease

sbatch -J test_real_t_so \
        -e $OUTBASE/logs/${TIME}/test/real/lamol_test_temporal.err \
        -o $OUTBASE/logs/${TIME}/test/real/lamol_test_temporal.log \
        --mem=15G \
        --partition=m40-short \
        --time=01:00:00 \
        --gres=gpu:1 \
         ./runtestreal.sh 0.25 6.25e-5 0 "so_t_1 so_t_2 so_t_3 so_t_4 so_t_5" so_data/so_labels $OUTBASE/models ~/Lamolrelease


sbatch -J test_ft_t_so \
        -e $OUTBASE/logs/${TIME}/test/finetune/lamol_test_temporal.err \
        -o $OUTBASE/logs/${TIME}/test/finetune/lamol_test_temporal.log \
        --mem=15G \
        --partition=m40-short \
        --time=01:00:00 \
        --gres=gpu:1 \
        ./runtest.sh 0.25 0.2 6.25e-5 0 "so_t_1 so_t_2 so_t_3 so_t_4 so_t_5" so_data/so_labels finetune $OUTBASE/models ~/Lamolrelease


## Train on whole so temporal train set, for getting a baseline
#sbatch -J test_whole_t_so \
#        -e $OUTBASE/logs/${TIME}/test/finetune/lamol_test_whole_so_temporal.err \
#        -o $OUTBASE/logs/${TIME}/test/finetune/lamol_test_whole_so_temporal.log \
#        --mem=15G \
#        --partition=m40-short \
#        --time=01:00:00 \
#        --gres=gpu:1 \
#         ./runtest.sh 0.25 0.2 6.25e-5 0 "so_t_all_1 so_t_all_2 so_t_all_3 so_t_all_4 so_t_all_5" so_data/so_labels finetune $OUTBASE/models/${TIME} ~/Lamolrelease
#
#for C in {1..10..9}; do
#  sbatch -J test_real_so_${C} \
#          -e $OUTBASE/logs/${TIME}/test/real/lamol_test_${C}.err \
#          -o $OUTBASE/logs/${TIME}/test/real/lamol_test_${C}.log \
#          --mem=15G \
#          --partition=m40-short \
#          --time=01:00:00 \
#          --gres=gpu:1 \
#          ./runtestreal.sh 0.25 6.25e-5 0 "so_${C}_1 so_${C}_2 so_${C}_3 so_${C}_4 so_${C}_5" so_data/so_labels $OUTBASE/models/${TIME} ~/Lamolrelease
#
#  sbatch -J test_ft_so_${C} \
#          -e $OUTBASE/logs/${TIME}/test/finetune/lamol_test_${C}.err \
#          -o $OUTBASE/logs/${TIME}/test/finetune/lamol_test_${C}.log \
#          --mem=15G \
#          --partition=m40-short \
#          --time=01:00:00 \
#          --gres=gpu:1 \
#          ./runtest.sh 0.25 0.2 6.25e-5 0 "so_${C}_1 so_${C}_2 so_${C}_3 so_${C}_4 so_${C}_5" so_data/so_labels finetune $OUTBASE/models/${TIME} ~/Lamolrelease
#
#  # Train on whole so train set, for getting a baseline
#  sbatch -J test_whole_so_${C} \
#          -e $OUTBASE/logs/${TIME}/test/finetune/lamol_test_whole_so_${C}.err \
#          -o $OUTBASE/logs/${TIME}/test/finetune/lamol_test_whole_so_${C}.log \
#          --mem=15G \
#          --partition=m40-short \
#          --time=01:00:00 \
#          --gres=gpu:1 \
#          ./runtest.sh 0.25 0.2 6.25e-5 0 "so_${C}_all_1 so_${C}_all_2 so_${C}_all_3 so_${C}_all_4 so_${C}_all_5" so_data/so_labels finetune $OUTBASE/models/${TIME} ~/Lamolrelease
#done