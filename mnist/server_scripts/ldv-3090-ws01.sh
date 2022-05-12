#!/bin/bash

########################################### cifar ###########################################
screen -d -S lle_iso_360 -m \
python3 /root/WorkSpace/essl_isometry/cifar10/lle_iso.py \
--gpu 0 --bsz=256 --num_workers=16 --lmbd=1 --fp16 \
--num_base_rot=8 --base_rot_range=360 --num_local_nn=8 --local_rot_range=10 --lmbd=1 \
--data_root /root/Datasets/cifar10/ \
--path_dir /root/WorkSpace/essl_isometry/cifar10/experiment/LLEIso360/

screen -d -S group_iso_180 -m \
python3 /root/WorkSpace/essl_isometry/cifar10/lle_iso.py \
--gpu 1 --bsz=256 --num_workers=16 --lmbd=1 --fp16 \
--num_base_rot=8 --base_rot_range=180 --num_local_nn=8 --local_rot_range=10 --lmbd=1 \
--data_root /root/Datasets/cifar10/ \
--path_dir /root/WorkSpace/essl_isometry/cifar10/experiment/LLEIso180/

screen -d -S cos_aug_pred -m \
python3 /root/WorkSpace/essl_isometry/cifar10/cos_aug_pred.py \
--gpu 0 --bsz=256 --num_workers=16 --lmbd=1 --fp16 --num_rot=8 --lmbd=1 \
--data_root /root/Datasets/cifar10/ \
--path_dir /root/WorkSpace/essl_isometry/cifar10/experiment/COSAugPred/

########################################### mnist ###########################################
screen -d -S local_iso_cos_sim -m \
python3 /root/WorkSpace/essl_isometry/mnist/local_iso_cos_sim.py \
--device cuda:0 --bsz=512 --num_workers=16 \
--num_base_rot=8 --base_rot_range=180 --num_nn=8 --local_rot_range=30 \
--data_path /root/Datasets/mnist_rotation_new/ \
--path_dir /root/WorkSpace/essl_isometry/mnist/experiment/LocalIsoCosSim/

screen -d -S global_iso_cos_sim -m \
python3 /root/WorkSpace/essl_isometry/mnist/global_iso_cos_sim.py \
--device cuda:0 --bsz=512 --num_workers=8 --fp16 \
--num_rot=8 --rot_range=180 \
--data_path /root/Datasets/mnist_rotation_new/ \
--path_dir /root/WorkSpace/essl_isometry/mnist/experiment/GlobalIsoCosSim/