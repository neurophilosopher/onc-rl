#!/bin/bash
#SBATCH --job-name=ppo_small_all
#SBATCH --output=logs/out_%j.txt
#SBATCH --error=logs/err_%j.txt
#SBATCH --time=12:00:00
#SBATCH --partition=all
#SBATCH --account=winter2026-comp579
#SBATCH --qos=comp579-1gpu-12h
#SBATCH --mem=16G

mkdir -p logs

source rl579/bin/activate

python run_baselines_ppo_small.py \
    --save-dir ./results_ppo_small \
    --envs invpend_full invpend_partial cheetah_full cheetah_partial swimmer_full \
    --timesteps 1000000
