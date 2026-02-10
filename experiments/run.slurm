#!/bin/bash
#SBATCH --job-name=run
#SBATCH --output=logs/run_%A_%a.out
#SBATCH --error=logs/run_%A_%a.err
#SBATCH --array=0-3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:59:59

CAMPAIGNS_YAML="runs/cifar-10-conv-med.yml"

# python standard_worker.py \
#   --campaigns-yaml "$CAMPAIGNS_YAML" \
#   --campaign-index "$SLURM_ARRAY_TASK_ID" \


for seed in {0..9}; do
    echo "Running campaign index: $SLURM_ARRAY_TASK_ID with seed: $seed"
    python standard_worker.py \
        --campaigns-yaml "$CAMPAIGNS_YAML" \
        --campaign-index "$SLURM_ARRAY_TASK_ID" \
        --override "seed=$seed" \
        --append "output=-$seed"
done
