#!/usr/bin/env bash


TIME=`(date +%Y-%m-%d-%H-%M-%S)`
NUM_EPS=15
LR=5e-4

OUTBASE=/mnt/nfs/scratch1/jpayan/Lamolrelease

mkdir -p $OUTBASE/logs/${TIME}/train/real
mkdir -p $OUTBASE/logs/${TIME}/train/finetune

#sbatch -J real_so \
#          -e $OUTBASE/logs/${TIME}/train/real/lamol_train.err \
#          -o $OUTBASE/logs/${TIME}/train/real/lamol_train.log \
#          --mem=15G \
#          --partition=m40-long \
#          --time=01-00:00:00 \
#          --gres=gpu:1 \
#          ./setupandrunrealexp.sh 0.25 $LR $NUM_EPS "so_1 so_2 so_3 so_4 so_5" so_data/so_labels \
#          $OUTBASE/models/${TIME} ~/Lamolrelease
#
#sbatch -J ft_so \
#        -e $OUTBASE/logs/${TIME}/train/finetune/lamol_train.err \
#        -o $OUTBASE/logs/${TIME}/train/finetune/lamol_train.log \
#        --mem=15G \
#        --partition=m40-long \
#        --time=01-00:00:00 \
#        --gres=gpu:1 \
#        ./setupandrunexp.sh 0.25 0.2 $LR $NUM_EPS "so_1 so_2 so_3 so_4 so_5" so_data/so_labels \
#        finetune $OUTBASE/models/${TIME} ~/Lamolrelease
#
## Train on whole so train set, for getting a baseline
#sbatch -J whole_so \
#        -e $OUTBASE/logs/${TIME}/train/finetune/lamol_train_whole_so.err \
#        -o $OUTBASE/logs/${TIME}/train/finetune/lamol_train_whole_so.log \
#        --mem=15G \
#        --partition=m40-long \
#        --time=01-00:00:00 \
#        --gres=gpu:1 \
#        ./setupandrunexp.sh 0.25 0.2 $LR $NUM_EPS "so_all_1 so_all_2 so_all_3 so_all_4 so_all_5" so_data/so_labels \
#        finetune $OUTBASE/models/${TIME} ~/Lamolrelease
#
#sbatch -J real_t_so \
#        -e $OUTBASE/logs/${TIME}/train/real/lamol_train_temporal.err \
#        -o $OUTBASE/logs/${TIME}/train/real/lamol_train_temporal.log \
#        --mem=15G \
#        --partition=m40-long \
#        --time=01-00:00:00 \
#        --gres=gpu:1 \
#        ./setupandrunrealexp.sh 0.25 $LR $NUM_EPS "so_t_1 so_t_2 so_t_3 so_t_4 so_t_5" so_data/so_labels \
#        $OUTBASE/models/${TIME} ~/Lamolrelease
#
#sbatch -J ft_t_so \
#        -e $OUTBASE/logs/${TIME}/train/finetune/lamol_train_temporal.err \
#        -o $OUTBASE/logs/${TIME}/train/finetune/lamol_train_temporal.log \
#        --mem=15G \
#        --partition=m40-long \
#        --time=01-00:00:00 \
#        --gres=gpu:1 \
#        ./setupandrunexp.sh 0.25 0.2 $LR $NUM_EPS "so_t_1 so_t_2 so_t_3 so_t_4 so_t_5" so_data/so_labels \
#        finetune $OUTBASE/models/${TIME} ~/Lamolrelease
#
## Train on whole so temporal train set, for getting a baseline
#sbatch -J whole_t_so \
#        -e $OUTBASE/logs/${TIME}/train/finetune/lamol_train_whole_so_temporal.err \
#        -o $OUTBASE/logs/${TIME}/train/finetune/lamol_train_whole_so_temporal.log \
#        --mem=15G \
#        --partition=m40-long \
#        --time=01-00:00:00 \
#        --gres=gpu:1 \
#        ./setupandrunexp.sh 0.25 0.2 $LR $NUM_EPS "so_t_all_1 so_t_all_2 so_t_all_3 so_t_all_4 so_t_all_5" so_data/so_labels \
#        finetune $OUTBASE/models/${TIME} ~/Lamolrelease
#
#for C in {1..10..9}; do
#  sbatch -J real_so_${C} \
#          -e $OUTBASE/logs/${TIME}/train/real/lamol_train_${C}.err \
#          -o $OUTBASE/logs/${TIME}/train/real/lamol_train_${C}.log \
#          --mem=15G \
#          --partition=m40-long \
#          --time=01-00:00:00 \
#          --gres=gpu:1 \
#          ./setupandrunrealexp.sh 0.25 $LR $NUM_EPS "so_${C}_1 so_${C}_2 so_${C}_3 so_${C}_4 so_${C}_5" so_data/so_labels \
#          $OUTBASE/models/${TIME} ~/Lamolrelease
#
#  sbatch -J ft_so_${C} \
#          -e $OUTBASE/logs/${TIME}/train/finetune/lamol_train_${C}.err \
#          -o $OUTBASE/logs/${TIME}/train/finetune/lamol_train_${C}.log \
#          --mem=15G \
#          --partition=m40-long \
#          --time=01-00:00:00 \
#          --gres=gpu:1 \
#          ./setupandrunexp.sh 0.25 0.2 $LR $NUM_EPS "so_${C}_1 so_${C}_2 so_${C}_3 so_${C}_4 so_${C}_5" so_data/so_labels \
#          finetune $OUTBASE/models/${TIME} ~/Lamolrelease
#
#  # Train on whole so train set, for getting a baseline
#  sbatch -J whole_so_${C} \
#          -e $OUTBASE/logs/${TIME}/train/finetune/lamol_train_whole_so_${C}.err \
#          -o $OUTBASE/logs/${TIME}/train/finetune/lamol_train_whole_so_${C}.log \
#          --mem=15G \
#          --partition=m40-long \
#          --time=01-00:00:00 \
#          --gres=gpu:1 \
#          ./setupandrunexp.sh 0.25 0.2 $LR $NUM_EPS "so_${C}_all_1 so_${C}_all_2 so_${C}_all_3 so_${C}_all_4 so_${C}_all_5" so_data/so_labels \
#          finetune $OUTBASE/models/${TIME} ~/Lamolrelease
#done

for k in 300 500 1000 1500 1800; do
  for seed in {0..9}; do
    sbatch -J gdumb_${k}_${seed} \
    -e $OUTBASE/logs/${TIME}/train/finetune/lamol_train_whole_gdumb_${k}_${seed}.err \
    -o $OUTBASE/logs/${TIME}/train/finetune/lamol_train_whole_gdumb_${k}_${seed}.log \
    --mem=15G \
    --partition=m40-long \
    --time=01-00:00:00 \
    --gres=gpu:1 \
    ./setupandrunexp.sh 0.25 0.2 $LR $NUM_EPS "gdumb_${k}_${seed}_1 gdumb_${k}_${seed}_2 gdumb_${k}_${seed}_3 gdumb_${k}_${seed}_4 gdumb_${k}_${seed}_5" so_data/so_labels \
    finetune $OUTBASE/models/${TIME} ~/Lamolrelease
  done
done