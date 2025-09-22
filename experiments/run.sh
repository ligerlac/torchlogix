# python train.py --dataset mnist --implementation python --batch-size 128 --eval-freq 50 --num-iterations 1000 --seed 1 -o results/baseline/1
# python train.py --dataset mnist --implementation python --batch-size 128 --eval-freq 50 --num-iterations 1000 --seed 2 -o results/baseline/2
# python train.py --dataset mnist --implementation python --batch-size 128 --eval-freq 50 --num-iterations 1000 --seed 3 -o results/baseline/3
# python train.py --dataset mnist --implementation python --batch-size 128 --eval-freq 50 --num-iterations 1000 --seed 4 -o results/alternative/1
# python train.py --dataset mnist --implementation python --batch-size 128 --eval-freq 50 --num-iterations 1000 --seed 5 -o results/alternative/2
# python train.py --dataset mnist --implementation python --batch-size 128 --eval-freq 50 --num-iterations 1000 --seed 6 -o results/alternative/3

# python train.py --dataset mnist --implementation python --batch-size 128 --eval-freq 50 --num-iterations 1000 --seed 1 --parametrization walsh -o results/walsh/1
# python train.py --dataset mnist --implementation python --batch-size 128 --eval-freq 50 --num-iterations 1000 --seed 2 --parametrization walsh -o results/walsh/2
# python train.py --dataset mnist --implementation python --batch-size 128 --eval-freq 50 --num-iterations 1000 --seed 3 --parametrization walsh -o results/walsh/3
# python train.py --dataset mnist --implementation python --batch-size 128 --eval-freq 50 --num-iterations 1000 --seed 4 --parametrization walsh -o results/walsh/4
# python train.py --dataset mnist --implementation python --batch-size 128 --eval-freq 50 --num-iterations 1000 --seed 5 --parametrization walsh -o results/walsh/5
# python train.py --dataset mnist --implementation python --batch-size 128 --eval-freq 50 --num-iterations 1000 --seed 6 --parametrization walsh -o results/walsh/6

python train.py --dataset mnist --implementation python --batch-size 128 --eval-freq 50 --num-iterations 1000 --seed 1 --parametrization walsh -o results/walsh-random/1
python train.py --dataset mnist --implementation python --batch-size 128 --eval-freq 50 --num-iterations 1000 --seed 2 --parametrization walsh -o results/walsh-random/2
python train.py --dataset mnist --implementation python --batch-size 128 --eval-freq 50 --num-iterations 1000 --seed 3 --parametrization walsh -o results/walsh-random/3
python train.py --dataset mnist --implementation python --batch-size 128 --eval-freq 50 --num-iterations 1000 --seed 4 --parametrization walsh -o results/walsh-random/4
python train.py --dataset mnist --implementation python --batch-size 128 --eval-freq 50 --num-iterations 1000 --seed 5 --parametrization walsh -o results/walsh-random/5
python train.py --dataset mnist --implementation python --batch-size 128 --eval-freq 50 --num-iterations 1000 --seed 6 --parametrization walsh -o results/walsh-random/6
