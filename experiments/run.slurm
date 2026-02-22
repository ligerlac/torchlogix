#!/bin/bash
#SBATCH --job-name=train
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --time=23:59:59
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --array=0-2   # number of runs - 1

# Define argument sets
ARGS=(
  "-a ClgnCifar10Small -o results/clgn-cifar-10-small"
  "-a ClgnCifar10Medium -o results/clgn-cifar-10-medium"
  "-a ClgnCifar10Large -o results/clgn-cifar-10-large"
)

# Run the command corresponding to this array index
python train.py --dataset cifar-10 --device cuda --compile-model -lr 0.02 ${ARGS[$SLURM_ARRAY_TASK_ID]} 
