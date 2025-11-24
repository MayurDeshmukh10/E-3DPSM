#!/bin/bash
#SBATCH -p gpu20
#SBATCH --gres gpu:1
#SBATCH --mem=244G
#SBATCH --signal=SIGUSR1@90
#SBATCH -o /CT/EventEgo3Dv2/work/code_variations/dp_att_lkf_lnes_update_deform_att/Final_results/out-%j.out
#SBATCH -e /CT/EventEgo3Dv2/work/code_variations/dp_att_lkf_lnes_update_deform_att/Final_results/err-%j.err
#SBATCH -t 3:00:00

cd /CT/EventEgo3Dv2/work/code_variations/dp_att_lkf_lnes_update_deform_att
  
source activate ee3dh100

# EXP_NAME='eval_att_finetuning_best_epoch_w_jitter_ts_10' TRAINING_TYPE='pretrain' BATCH_SIZE=9 srun python3 run.py test --config ./configs/evaluate_finetune_preprocessed_input_attention_kf_lnes.yaml
EXP_NAME='eval_non_causal_epoch_24_finetune' TRAINING_TYPE='pretrain' BATCH_SIZE=9 srun python3 run.py test --config ./configs/evaluate_deform_attention_kf_lnes.yaml

# EXP_NAME='eval_non_causal_epoch_24_finetune' TRAINING_TYPE='pretrain' BATCH_SIZE=9 python3 run.py test --config ./configs/evaluate_preprocessed_input_attention_kf_lnes_temp.yaml
