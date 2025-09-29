# python3 train.py --dataset mnist20x20 --implementation python --batch-size 32 --eval-freq 200 --num-iterations 1000 --seed 1 --parametrization walsh --walsh-sampling sigmoid -o results/walsh-sigmoid/1

# python3 train.py --dataset mnist20x20 --implementation python --batch-size 32 --eval-freq 200 --num-iterations 1000 --seed 1 --parametrization walsh --walsh-sampling softmax -o results/walsh-softmax/1

#python3 train.py --dataset mnist20x20 --implementation python --batch-size 32 --eval-freq 500 --num-iterations 1000 --seed 1 --parametrization walsh --forward-sampling gumbel_hard -o results/walsh-gumbel_hard/1

# python3 train.py --dataset mnist20x20 --implementation python --batch-size 32 --eval-freq 500 --num-iterations 1000 --seed 1 --parametrization walsh --walsh-sampling gumbel_soft -o results/walsh-gumbel_soft/1

#python3 train.py --dataset mnist20x20 --implementation python --batch-size 32 --eval-freq 500 --num-iterations 1000 --seed 1 --parametrization raw --forward-sampling gumbel_hard -o results/raw-gumbel_hard/1

python3 train.py --dataset mnist20x20 --implementation python --batch-size 32 --eval-freq 500 --num-iterations 1000 --seed 1 --parametrization raw --forward-sampling gumbel_soft -o results/raw-gumbel_soft/1

#python3 train.py --dataset mnist20x20 --implementation python --batch-size 32 --eval-freq 500 --num-iterations 1000 --seed 1 --parametrization raw --forward-sampling sigmoid -o results/raw-sigmoid/1
