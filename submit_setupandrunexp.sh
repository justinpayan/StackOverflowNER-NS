#!/usr/bin/env bash


TIME=`(date +%Y-%m-%d-%H-%M-%S)`

OUTBASE=/mnt/nfs/scratch1/jpayan/Lamolrelease

mkdir -p $OUTBASE/logs/${TIME}/train/lll
mkdir -p $OUTBASE/logs/${TIME}/train/real
mkdir -p $OUTBASE/logs/${TIME}/train/finetune

sbatch -J real_so \
          -e $OUTBASE/logs/${TIME}/train/real/lamol_train.err \
          -o $OUTBASE/logs/${TIME}/train/real/lamol_train.log \
          --mem=15G \
          --partition=m40-long \
          --time=01-00:00:00 \
          --gres=gpu:1 \
          ./setupandrunrealexp.sh 0.25 6.25e-5 0 "so_1 so_2 so_3 so_4 so_5" so_data/so_labels \
          $OUTBASE/models/${TIME} ~/Lamolrelease

sbatch -J ft_so \
        -e $OUTBASE/logs/${TIME}/train/finetune/lamol_train.err \
        -o $OUTBASE/logs/${TIME}/train/finetune/lamol_train.log \
        --mem=15G \
        --partition=m40-long \
        --time=01-00:00:00 \
        --gres=gpu:1 \
        ./setupandrunexp.sh 0.25 0.2 6.25e-5 0 "so_1 so_2 so_3 so_4 so_5" so_data/so_labels \
        finetune $OUTBASE/models/${TIME} ~/Lamolrelease

# Train on whole so train set, for getting a baseline
sbatch -J whole_so \
        -e $OUTBASE/logs/${TIME}/train/finetune/lamol_train_whole_so.err \
        -o $OUTBASE/logs/${TIME}/train/finetune/lamol_train_whole_so.log \
        --mem=15G \
        --partition=m40-long \
        --time=01-00:00:00 \
        --gres=gpu:1 \
        ./setupandrunexp.sh 0.25 0.2 6.25e-5 0 "so_all_1 so_all_2 so_all_3 so_all_4 so_all_5" so_data/so_labels \
        finetune $OUTBASE/models/${TIME} ~/Lamolrelease

sbatch -J real_t_so \
        -e $OUTBASE/logs/${TIME}/train/real/lamol_train_temporal.err \
        -o $OUTBASE/logs/${TIME}/train/real/lamol_train_temporal.log \
        --mem=15G \
        --partition=m40-long \
        --time=01-00:00:00 \
        --gres=gpu:1 \
        ./setupandrunrealexp.sh 0.25 6.25e-5 0 "so_t_1 so_t_2 so_t_3 so_t_4 so_t_5" so_data/so_labels \
        $OUTBASE/models/${TIME} ~/Lamolrelease

sbatch -J ft_t_so \
        -e $OUTBASE/logs/${TIME}/train/finetune/lamol_train_temporal.err \
        -o $OUTBASE/logs/${TIME}/train/finetune/lamol_train_temporal.log \
        --mem=15G \
        --partition=m40-long \
        --time=01-00:00:00 \
        --gres=gpu:1 \
        ./setupandrunexp.sh 0.25 0.2 6.25e-5 0 "so_t_1 so_t_2 so_t_3 so_t_4 so_t_5" so_data/so_labels \
        finetune $OUTBASE/models/${TIME} ~/Lamolrelease

# Train on whole so temporal train set, for getting a baseline
sbatch -J whole_t_so \
        -e $OUTBASE/logs/${TIME}/train/finetune/lamol_train_whole_so_temporal.err \
        -o $OUTBASE/logs/${TIME}/train/finetune/lamol_train_whole_so_temporal.log \
        --mem=15G \
        --partition=m40-long \
        --time=01-00:00:00 \
        --gres=gpu:1 \
        ./setupandrunexp.sh 0.25 0.2 6.25e-5 0 "so_t_all_1 so_t_all_2 so_t_all_3 so_t_all_4 so_t_all_5" so_data/so_labels \
        finetune $OUTBASE/models/${TIME} ~/Lamolrelease

for C in {1..10..9}; do
  sbatch -J real_so_${C} \
          -e $OUTBASE/logs/${TIME}/train/real/lamol_train_${C}.err \
          -o $OUTBASE/logs/${TIME}/train/real/lamol_train_${C}.log \
          --mem=15G \
          --partition=m40-long \
          --time=01-00:00:00 \
          --gres=gpu:1 \
          ./setupandrunrealexp.sh 0.25 6.25e-5 0 "so_${C}_1 so_${C}_2 so_${C}_3 so_${C}_4 so_${C}_5" so_data/so_labels \
          $OUTBASE/models/${TIME} ~/Lamolrelease

  sbatch -J ft_so_${C} \
          -e $OUTBASE/logs/${TIME}/train/finetune/lamol_train_${C}.err \
          -o $OUTBASE/logs/${TIME}/train/finetune/lamol_train_${C}.log \
          --mem=15G \
          --partition=m40-long \
          --time=01-00:00:00 \
          --gres=gpu:1 \
          ./setupandrunexp.sh 0.25 0.2 6.25e-5 0 "so_${C}_1 so_${C}_2 so_${C}_3 so_${C}_4 so_${C}_5" so_data/so_labels \
          finetune $OUTBASE/models/${TIME} ~/Lamolrelease

  # Train on whole so train set, for getting a baseline
  sbatch -J whole_so_${C} \
          -e $OUTBASE/logs/${TIME}/train/finetune/lamol_train_whole_so_${C}.err \
          -o $OUTBASE/logs/${TIME}/train/finetune/lamol_train_whole_so_${C}.log \
          --mem=15G \
          --partition=m40-long \
          --time=01-00:00:00 \
          --gres=gpu:1 \
          ./setupandrunexp.sh 0.25 0.2 6.25e-5 0 "so_${C}_all_1 so_${C}_all_2 so_${C}_all_3 so_${C}_all_4 so_${C}_all_5" so_data/so_labels \
          finetune $OUTBASE/models/${TIME} ~/Lamolrelease
done