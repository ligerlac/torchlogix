#!/bin/bash
#SBATCH --job-name=hpo_optuna
#SBATCH --output=logs/hpo_%A_%a.out
#SBATCH --error=logs/hpo_%A_%a.err
#SBATCH --array=0-6
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=23:59:59

# Point to your shared database
STORAGE="sqlite:///${OPTUNA_DB_PATH}"

CAMPAIGNS_YAML="campaigns/cifar-10-conv-small.yml"

# Verify database is accessible from this node
if [ ! -f $OPTUNA_DB_PATH ]; then
    echo "ERROR: Cannot access database from node $(hostname)"
    exit 1
fi

python hpo_worker.py \
  --storage "$STORAGE" \
  --campaigns-yaml "$CAMPAIGNS_YAML" \
  --campaign-index "$SLURM_ARRAY_TASK_ID" \
  --n-trials 200
