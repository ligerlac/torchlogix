#!/bin/bash

# CAMPAIGNS_YAML="runs/cifar-10-conv-binarization.yml"
# CAMPAIGNS_YAML="runs/jsc-binarization.yml"
# CAMPAIGNS_YAML="runs/jsc-binarization-new.yml"
CAMPAIGNS_YAML="runs/cifar-10-conv-res-prob.yml"


# # for i in {0..29}
# for i in {0..39}
# do
#   echo "Starting Run for index $i"
#   python standard_worker.py \
#     --campaigns-yaml "$CAMPAIGNS_YAML" \
#     --campaign-index "$i"
# done


for i in {0..23}; do
  for seed in {0..1}; do
    echo "Running campaign index: $i"
    python standard_worker.py \
        --campaigns-yaml "$CAMPAIGNS_YAML" \
        --campaign-index "$i" \
        --override "seed=$seed" \
        --append "output=-$seed"
  done
done
