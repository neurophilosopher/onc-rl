#!/bin/bash
#SBATCH --job-name=ars_all
#SBATCH --output=logs/out_%j.txt
#SBATCH --error=logs/err_%j.txt
#SBATCH --time=12:00:00
#SBATCH --partition=all
#SBATCH --account=winter2026-comp579
#SBATCH --mem=16G

mkdir -p logs

source rl579/bin/activate

python run_ars.py \
    --save-dir ./results_ars \
    --envs invpend_full invpend_partial cheetah_full cheetah_partial swimmer_full \
    --steps 25000
