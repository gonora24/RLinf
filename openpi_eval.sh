#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --time=10
#SBATCH --gres=gpu:1 
#SBATCH --job-name=openpi_eval

source openpi-venv/bin/activate


bash examples/embodiment/eval_embodiment.sh libero_goal_grpo_openpi