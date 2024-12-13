#!/bin/bash
#SBATCH --signal=B:SIGTERM@120
#SBATCH -p gpu20
#SBATCH --mem=192G
#SBATCH --gres gpu:1
#SBATCH -o ./logs/slurm_evaluations/evalutate-%j.out
#SBATCH -t 24:00:00

cd /CT/EventEgo3Dv2/work/EventEgo3D

eval "$(conda shell.bash hook)"
  
source activate EE3D

export BATCH_SIZE=27
# export TEMPORAL_STEPS="20"

python evaluate_ee3d_s.py
