#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1 
#SBATCH --job-name=object_dsrl_openpi

export WANDB_API_KEY='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'
export WANDB_EMAIL='noragorhan@gmail.com'
export WANDB_USERNAME='noragorhan'
export WANDB_TEAM='noras-masterarbeit'
export WANDB_RUN_GROUP='DSRL'
export WANDB_JOB_TYPE='train'
export WANDB_MODE='offline'

export DEBUG=0
export ARVIND=0

# Set absolute repo path (for sbatch compatibility)
if [ "$ARVIND" == 0 ]; then
    export REPO_PATH="/home/ka/ka_anthropomatik/ka_eu3660/projects/RLinf"
else
    export REPO_PATH="/home/ka/ka_anthropomatik/ka_db6855/nora/RLinf"
fi

export ABS_LOG_DIR="/pfs/work9/workspace/scratch/ka_eu3660-rlinf_tmp"

module load devel/miniforge
module load devel/cuda

source openpi-venv/bin/activate

bash examples/embodiment/run_embodiment.sh libero_object_dsrl_openpi