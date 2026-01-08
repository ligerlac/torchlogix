#!/bin/bash
#SBATCH --job-name=hpo_optuna
#SBATCH --output=logs/hpo_%A_%a.out
#SBATCH --error=logs/hpo_%A_%a.err
#SBATCH --array=0-5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00

# Point to your shared database
STORAGE="sqlite:///${OPTUNA_DB_PATH}"
CAMPAIGNS_YAML="campaigns/cifar-10-conv.yml"

# Verify database is accessible from this node
if [ ! -f $OPTUNA_DB_PATH ]; then
    echo "ERROR: Cannot access database from node $(hostname)"
    exit 1
fi

python hpo_worker.py \
  --storage "$STORAGE" \
  --campaigns-yaml "$CAMPAIGNS_YAML" \
  --campaign-index "$SLURM_ARRAY_TASK_ID" \
  --n-trials 15
