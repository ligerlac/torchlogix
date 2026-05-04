DATASET="mnist"
NUM_ITERATIONS="100"
EVAL_FREQ="50"
LEARNING_RATE="0.01"
LUT_RANK="4"
PARAMETRIZATION="warp"
# Classifcation
LOSS="cross_entropy"
EVAL_FUNCTIONS=("loss" "acc")
ARCHITECTURE="DlgnMnistSmall"

python train.py \
    --dataset $DATASET \
    --num-iterations $NUM_ITERATIONS \
    --eval-freq $EVAL_FREQ \
    --loss $LOSS \
    --eval-functions ${EVAL_FUNCTIONS[@]} \
    --learning-rate $LEARNING_RATE \
    --lut-rank $LUT_RANK \
    --parametrization $PARAMETRIZATION \
    --architecture $ARCHITECTURE

# Regression
LOSS="mse"
EVAL_FUNCTIONS=("loss")
ARCHITECTURE="DlgnRegAffineMnistSmall"

python train.py \
    --dataset $DATASET \
    --num-iterations $NUM_ITERATIONS \
    --eval-freq $EVAL_FREQ \
    --loss $LOSS \
    --eval-functions ${EVAL_FUNCTIONS[@]} \
    --learning-rate $LEARNING_RATE \
    --lut-rank $LUT_RANK \
    --parametrization $PARAMETRIZATION \
    --architecture $ARCHITECTURE