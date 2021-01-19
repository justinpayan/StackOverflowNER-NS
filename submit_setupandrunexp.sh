#!/usr/bin/env bash

OUTBASE=/iesl/canvas/jpayan/Lamolrelease

sbatch -J train_so \
        -e $OUTBASE/logs/train/lamol_train.err \
        -o $OUTBASE/logs/train/lamol_train.log \
        --mem=15G \
        --partition=gpu \
        --time=01-00:00:00 \
        --gres=gpu:1 \
        ./setupandrunexp.sh 0.25 0.2 6.25e-5 0 "so_1 so_2 so_3 so_4 so_5" so_data/so_labels \
        lll $OUTBASE/models ~/Lamolrelease $OUTBASE