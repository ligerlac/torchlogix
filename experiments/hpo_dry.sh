#!/bin/bash

# Point to your shared database
STORAGE="sqlite:///${OPTUNA_DB_PATH}"
CAMPAIGNS_YAML="campaigns/parametrization-dense.yml"


python hpo_worker.py \
  --storage "$STORAGE" \
  --campaigns-yaml "$CAMPAIGNS_YAML" \
  --campaign-index 0 \
  --n-trials 1

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

python hpo_worker.py \
  --storage "$STORAGE" \
  --campaigns-yaml "$CAMPAIGNS_YAML" \
  --campaign-index 6 \
  --n-trials 1

python hpo_worker.py \
  --storage "$STORAGE" \
  --campaigns-yaml "$CAMPAIGNS_YAML" \
  --campaign-index 7 \
  --n-trials 1

python hpo_worker.py \
  --storage "$STORAGE" \
  --campaigns-yaml "$CAMPAIGNS_YAML" \
  --campaign-index 8 \
  --n-trials 1

python hpo_worker.py \
  --storage "$STORAGE" \
  --campaigns-yaml "$CAMPAIGNS_YAML" \
  --campaign-index 9 \
  --n-trials 1

python hpo_worker.py \
  --storage "$STORAGE" \
  --campaigns-yaml "$CAMPAIGNS_YAML" \
  --campaign-index 10 \
  --n-trials 1

python hpo_worker.py \
  --storage "$STORAGE" \
  --campaigns-yaml "$CAMPAIGNS_YAML" \
  --campaign-index 11 \
  --n-trials 1
