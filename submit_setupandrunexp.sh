#!/usr/bin/env bash

OUTBASE=/mnt/nfs/scratch1/jpayan/Lamolrelease

mkdir -p $OUTBASE/logs/train/lll
mkdir -p $OUTBASE/logs/train/finetune

#sbatch -J lll_so \
#        -e $OUTBASE/logs/train/lll/lamol_train.err \
#        -o $OUTBASE/logs/train/lll/lamol_train.log \
#        --mem=15G \
#        --partition=m40-long \
#        --time=01-00:00:00 \
#        --gres=gpu:1 \
#        ./setupandrunexp.sh 0.25 0.2 6.25e-5 0 "so_1 so_2 so_3 so_4 so_5" so_data/so_labels \
#        lll $OUTBASE/models ~/Lamolrelease

sbatch -J real_so \
        -e $OUTBASE/logs/train/real/lamol_train.err \
        -o $OUTBASE/logs/train/real/lamol_train.log \
        --mem=15G \
        --partition=m40-long \
        --time=01-00:00:00 \
        --gres=gpu:1 \
        ./setupandrunrealexp.sh 0.25 6.25e-5 0 "so_1 so_2 so_3 so_4 so_5" so_data/so_labels \
        $OUTBASE/models ~/Lamolrelease

sbatch -J ft_so \
        -e $OUTBASE/logs/train/finetune/lamol_train.err \
        -o $OUTBASE/logs/train/finetune/lamol_train.log \
        --mem=15G \
        --partition=m40-long \
        --time=01-00:00:00 \
        --gres=gpu:1 \
        ./setupandrunexp.sh 0.25 0.2 6.25e-5 0 "so_1 so_2 so_3 so_4 so_5" so_data/so_labels \
        finetune $OUTBASE/models ~/Lamolrelease