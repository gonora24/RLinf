#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1 
#SBATCH --job-name=ev_dsrl_meta_pi05_eval

export WANDB_API_KEY='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'
export WANDB_EMAIL='noragorhan@gmail.com'
export WANDB_USERNAME='noragorhan'
export WANDB_TEAM='noras-masterarbeit'
export WANDB_RUN_GROUP='DSRL'
export WANDB_JOB_TYPE='eval'

export DEBUG=0

module load devel/miniforge
module load devel/cuda

source openpi-meta-venv/bin/activate

bash examples/embodiment/eval_embodiment.sh metaworld_50_dsrl_openpi_pi05 #metaworld_50_dsrl_openpi_pi05 #libero_spatial_dsrl_openpi #libero_90_dsrl_openpi #libero_90_noise_sac_openpi05 #maniskill_ppo_openpi_pi05 #libero_goal_ppo_openpi_pi05_resume