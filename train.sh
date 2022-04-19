#!/usr/bin/env bash
MVS_TRAINING="datasets/mvs_training/dtu"
python train.py --dataset=dtu_yao --batch_size=1 --trainpath=$MVS_TRAINING --trainlist lists/dtu/train.txt --testlist lists/dtu/test.txt --numdepth=192 --logdir ./checkpoints/wtf $@
