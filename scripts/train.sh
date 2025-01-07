#!/bin/bash
#SBATCH -p gpu22
#SBATCH --gres gpu:a100:1
#SBATCH -o outputs/out-%j.out
#SBATCH -e error/err-%j.err
#SBATCH -t 24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mdeshmuk@mpi-inf.mpg.de

cd /CT/EventEgo3Dv2/work/EventEgo3Dv2
  
source activate ee3dh100
                 
# CHECKPOINT_PATH='' CUDA_VISIBLE_DEVICES=0 BATCH_SIZE=9 python train.py
EXP_NAME='ssm_trail_1' CHECKPOINT_PATH='/CT/EventEgo3Dv2/work/EventEgo3Dv2/logs/output/EgoEvent/EgoHPE/EgoEvent_2025-01-06-12-36_ssm_trail_1_epoch_5_checkpoint.pth' CUDA_VISIBLE_DEVICES=0 BATCH_SIZE=10 python train.py
# CHECKPOINT_PATH='/CT/EventEgo3Dv2/work/EventEgo3Dv2/logs/output/EgoEvent/EgoHPE/EgoEvent_2024-11-21-16-44_epoch_29_checkpoint.pth' CUDA_VISIBLE_DEVICES=0 BATCH_SIZE=6 python train.py