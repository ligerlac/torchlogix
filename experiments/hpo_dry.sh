#!/bin/bash

# Point to your shared database
STORAGE="sqlite:///${OPTUNA_DB_PATH}"
CAMPAIGNS_YAML="campaigns/cifar-10-conv.yml"


python hpo_worker.py \
  --storage "$STORAGE" \
  --campaigns-yaml "$CAMPAIGNS_YAML" \
  --campaign-index 0 \
  --n-trials 2

python hpo_worker.py \
  --storage "$STORAGE" \
  --campaigns-yaml "$CAMPAIGNS_YAML" \
  --campaign-index 1 \
  --n-trials 1

python hpo_worker.py \
  --storage "$STORAGE" \
  --campaigns-yaml "$CAMPAIGNS_YAML" \
  --campaign-index 2 \
  --n-trials 1

python hpo_worker.py \
  --storage "$STORAGE" \
  --campaigns-yaml "$CAMPAIGNS_YAML" \
  --campaign-index 3 \
  --n-trials 1

python hpo_worker.py \
  --storage "$STORAGE" \
  --campaigns-yaml "$CAMPAIGNS_YAML" \
  --campaign-index 4 \
  --n-trials 1

python hpo_worker.py \
  --storage "$STORAGE" \
  --campaigns-yaml "$CAMPAIGNS_YAML" \
  --campaign-index 5 \
  --n-trials 1

