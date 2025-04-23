#!/bin/bash
#SBATCH -p gpu20
#SBATCH --gres gpu:1
#SBATCH --mem=244G
#SBATCH --signal=SIGUSR1@90
#SBATCH -o new_outputs/out-%j.out
#SBATCH -e new_error/err-%j.err
#SBATCH -t 3:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mdeshmuk@mpi-inf.mpg.dede

cd /CT/EventEgo3Dv2/work/code_variations/dp_att_lkf_lnes
  
source activate ee3dh100

# EXP_NAME='eval_att_finetuning_best_epoch_w_jitter_ts_10' TRAINING_TYPE='pretrain' BATCH_SIZE=9 srun python3 run.py test --config ./configs/evaluate_finetune_preprocessed_input_attention_kf_lnes.yaml
EXP_NAME='eval_att_finetuning_best_epoch_w_jitter_ts_10' TRAINING_TYPE='pretrain' BATCH_SIZE=9 srun python3 run.py test --config ./configs/test.yaml
