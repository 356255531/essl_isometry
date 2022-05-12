#!/bin/bash

#COMMAND="
#docker exec ga63vuf \
#python3 /root/WorkSpace/essl_isometry/mnist/local_iso_cos_sim_contrastive_uniform_single.py \
#--device cuda:0 --bsz=32 --num_workers=32 --lmbd=0.1 \
#--num_base_rot=8 --base_rot_range=360 --num_nn=8 --local_rot_range=80 \
#--data_path /root/Datasets/mnist_rotation_new/ \
#--path_dir /root/WorkSpace/essl_isometry/mnist/experiment/LocalIsoCosSimContrastUniform/
#"

#COMMAND="
#docker exec ga63vuf \
#python3 /root/WorkSpace/essl_isometry/mnist/local_iso_cos_sim_contrastive_uniform.py \
#--device cuda:0 --bsz=32 --num_workers=32 --lmbd=0.1 \
#--num_base_rot=8 --base_rot_range=360 --num_nn=8 --local_rot_range=80 \
#--data_path /root/Datasets/mnist_rotation_new/ \
#--path_dir /root/WorkSpace/essl_isometry/mnist/experiment/LocalIsoCosSimContrastUniform/
#"

#COMMAND="
#docker exec ga63vuf \
#python3 /root/WorkSpace/essl_isometry/mnist/local_iso_cos_sim_contrastive_random.py \
#--device cuda:1 --bsz=32 --num_workers=32 --lmbd=0.1 \
#--num_base_rot=8 --base_rot_range=360 --num_nn=8 --local_rot_range=80 \
#--data_path /root/Datasets/mnist_rotation_new/ \
#--path_dir /root/WorkSpace/essl_isometry/mnist/experiment/LocalIsoCosSimContrastRandom/
#"

COMMAND="
docker exec ga63vuf \
python3 /root/WorkSpace/essl_isometry/mnist/local_iso_cos_sim_uniform.py \
--device cuda:0 --bsz=64 --num_workers=32 --fp16 \
--num_base_rot=36 --base_rot_range=360 --num_nn=8 --local_rot_range=90 \
--data_path /root/Datasets/mnist_rotation_new/ \
--path_dir /root/WorkSpace/essl_isometry/mnist/experiment/LocalIsoCosSimUniform/
"

#COMMAND="
#docker exec ga63vuf \
#python3 /root/WorkSpace/essl_isometry/mnist/local_iso_cos_sim_random.py \
#--device cuda:1 --bsz=128 --num_workers=32 --fp16 \
#--num_base_rot=36 --base_rot_range=360 --num_nn=8 --local_rot_range=20 \
#--data_path /root/Datasets/mnist_rotation_new/ \
#--path_dir /root/WorkSpace/essl_isometry/mnist/experiment/LocalIsoCosSim/
#"

#COMMAND="
#docker exec ga63vuf \
#python3 /root/WorkSpace/essl_isometry/mnist/global_iso_cos_sim.py \
#--device cuda:0 --bsz=512 --num_workers=32 --fp16 \
#--num_rot=8 --rot_range=360 \
#--data_path /root/Datasets/mnist_rotation_new/ \
#--path_dir /root/WorkSpace/essl_isometry/mnist/experiment/GlobalIsoCosSim/
#"

# ldv-3090-ws01
USER="ga63vuf"
HOST="ldv-3090-ws01"
scp ../*.py ${USER}@${HOST}:/nas/netstore/ldv/ga63vuf/WorkSpace/essl_isometry/mnist/
ssh -l ${USER} ${HOST} ${COMMAND}

