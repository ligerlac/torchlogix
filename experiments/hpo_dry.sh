#!/bin/bash

# Point to your shared database
# CAMPAIGNS_YAML="campaigns/parametrization-dense.yml"
# CAMPAIGNS_YAML="campaigns/dwn-configs.yml"
# CAMPAIGNS_YAML="campaigns/lutrank-gumbel.yml"
# CAMPAIGNS_YAML="campaigns/cifar-10-conv-small.yml"
# CAMPAIGNS_YAML="campaigns/conv-test.yml"
# CAMPAIGNS_YAML="campaigns/dwn-configs.yml"
# CAMPAIGNS_YAML="campaigns/cifar-10-binarization.yml"
CAMPAIGNS_YAML="campaigns/cifar-10-dense-deep.yml"

# Storage URL for Optuna
STORAGE="sqlite:///${OPTUNA_DB_PATH}"

# for i in {0..13}
for i in {0..2}
do
  echo "Starting HPO worker for campaign index $i"
  python hpo_worker.py \
    --storage "$STORAGE" \
    --campaigns-yaml "$CAMPAIGNS_YAML" \
    --campaign-index "$i" \
    --n-trials 1
done
