#!/bin/bash

# ldv-3090-ws01 has two 3090 gpus
screen -d -S lle_iso_360 -m \
python3 /root/WorkSpace/essl_isometry/cifar10/lle_iso.py \
--gpu 0 --bsz=128 --num_workers=16 --lmbd=1 --fp16 \
--num_base_rot=8 --base_rot_range=360 --num_local_nn=8 --local_rot_range=10 --lmbd=1 \
--data_root /root/Datasets/cifar10/ \
--path_dir /root/WorkSpace/essl_isometry/cifar10/experiment/LLEIso360/

screen -d -S group_iso_180 -m \
python3 /root/WorkSpace/essl_isometry/cifar10/lle_iso.py \
--gpu 1 --bsz=128 --num_workers=16 --lmbd=1 --fp16 \
--num_base_rot=8 --base_rot_range=180 --num_local_nn=8 --local_rot_range=10 --lmbd=1 \
--data_root /root/Datasets/cifar10/ \
--path_dir /root/WorkSpace/essl_isometry/cifar10/experiment/LLEIso180/
