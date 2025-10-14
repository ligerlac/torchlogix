# For the money plot in the intro: Large dense network on CIFAR-10 with 5 thresholds
# Baseline: raw parametrization, soft sampling, residual weight init
# Ours: walsh parametrization, gumbel_soft sampling, residual weight init

python train.py \
    --dataset cifar-10-5-thresholds \
    --architecture DlgnCifar10Large \
    --parametrization raw \
    --implementation python \
    --device cuda \
    --batch-size 64 \
    --eval-freq 500 \
    --num-iterations 50_000 \
    --seed 0 \
    --forward-sampling soft \
    --weight-init residual \
    --output results/dlgn-baseline

python train.py \
    --dataset cifar-10-5-thresholds \
    --architecture DlgnCifar10Large \
    --parametrization raw \
    --implementation python \
    --device cuda \
    --batch-size 64 \
    --eval-freq 500 \
    --num-iterations 50_000 \
    --seed 0 \
    --forward-sampling gumbel_soft \
    --weight-init residual \
    --output results/dlgn-ours


# Comparing weight initializations: Small conv net on CIFAR-10 with 3 thresholds
# Baseline: raw parametrization, soft sampling, random & residual weight init
# Ours: walsh parametrization, gumbel_soft sampling, random & residual weight init

python train.py \
    --dataset cifar-10-3-thresholds \
    --architecture ClgnCifar10SmallRes \
    --parametrization raw \
    --implementation python \
    --device cuda \
    --batch-size 64 \
    --eval-freq 500 \
    --num-iterations 50_000 \
    --seed 0 \
    --forward-sampling soft \
    --weight-init random \
    --output results/clgn-baseline-random/

python train.py \
    --dataset cifar-10-3-thresholds \
    --architecture ClgnCifar10SmallRes \
    --parametrization raw \
    --implementation python \
    --device cuda \
    --batch-size 64 \
    --eval-freq 500 \
    --num-iterations 50_000 \
    --seed 0 \
    --forward-sampling soft \
    --weight-init residual \
    --output results/clgn-baseline-residual/

python train.py \
    --dataset cifar-10-3-thresholds \
    --architecture ClgnCifar10SmallRes \
    --parametrization walsh \
    --implementation python \
    --device cuda \
    --batch-size 64 \
    --eval-freq 500 \
    --num-iterations 50_000 \
    --seed 0 \
    --forward-sampling gumbel_soft \
    --weight-init random \
    --output results/clgn-ours-random/

python train.py \
    --dataset cifar-10-3-thresholds \
    --architecture ClgnCifar10SmallRes \
    --parametrization walsh \
    --implementation python \
    --device cuda \
    --batch-size 64 \
    --eval-freq 500 \
    --num-iterations 50_000 \
    --seed 0 \
    --forward-sampling gumbel_soft \
    --weight-init residual \
    --output results/clgn-ours-residual/
