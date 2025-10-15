# ⚠️ For the paper "WARP-LUTs - Walsh-Assisted Relaxation for Probabilistic Look Up Tables" ⚠️ 

You can run the experiments for shown in the paper like so
```
sh run.sh
```
And make the plots:
```
python make-paper-plots.py
```

# TorchLogix Experiments

This directory contains scripts for training, evaluating, and compiling TorchLogix models.

## Workflow

The experiment workflow is split into three separate, sequential scripts:

### 1. Training (`train.py`)

Train a TorchLogix model on a dataset:

```bash
python train.py --dataset mnist --num-iterations 100000 --output results/mnist_run1/
```

**Outputs:**
- `best_model.pt` - Best model based on validation accuracy
- `final_model.pt` - Final model after all iterations
- `training_metrics.csv` - Training and validation metrics over time
- `training_config.json` - Configuration used for training

**Key Arguments:**
- `--dataset`: Dataset to train on (mnist, cifar-10-3-thresholds, adult, etc.)
- `--architecture`: Model architecture (randomly_connected, cnn, fully_connected)
- `--num-neurons`, `--num-layers`: Model size parameters
- `--num-iterations`, `--eval-freq`: Training schedule
- `--implementation`: cuda (fast) or python (slow, CPU-only)

### 2. Evaluation (`evaluate.py`)

Evaluate a trained model with comprehensive metrics:

```bash
python evaluate.py --model-path results/mnist_run1/best_model.pt --include-test --detailed
```

**Outputs:**
- `evaluation_results.json` - Complete evaluation results
- `evaluation_summary.csv` - Summary metrics in CSV format
- `*_classification_report.json` - Per-class metrics (if --detailed)
- `*_confusion_matrix.csv` - Confusion matrices (if --detailed)

**Key Arguments:**
- `--model-path`: Path to trained model (.pt file)
- `--include-train`, `--include-test`: Include train/test sets in evaluation
- `--train-mode`: Also evaluate in training mode (differentiable)
- `--detailed`: Generate classification reports and confusion matrices
- `--packbits-eval`: Test PackBits inference (CUDA only)

### 3. Compilation (`compile.py`)

Compile trained models to optimized C code:

```bash
python compile.py --model-path results/mnist_run1/best_model.pt --benchmark-original
```

**Outputs:**
- `compilation_summary.json` - Overall compilation results
- `compilation_summary.csv` - Summary metrics
- `opt*_bits*/` - Directories for each optimization configuration
  - `model_opt*_*bits.so` - Compiled shared library
  - `compilation_results.json` - Results for this configuration

**Key Arguments:**
- `--model-path`: Path to trained model
- `--opt-levels`: Optimization levels to test (default: 0,1,2,3)
- `--bit-counts`: Bit counts to test (default: 32,64)
- `--benchmark-original`: Also benchmark PyTorch model
- `--compiler`: C compiler (gcc or clang)

## Example Complete Workflow

```bash
# 1. Train a model
python train.py \
  --dataset mnist \
  --architecture randomly_connected \
  --num-neurons 64000 \
  --num-layers 6 \
  --num-iterations 200000 \
  --output results/mnist_large/

# 2. Evaluate the trained model
python evaluate.py \
  --model-path results/mnist_large/best_model.pt \
  --include-test \
  --train-mode \
  --detailed \
  --packbits-eval \
  --output results/mnist_large/evaluation/

# 3. Compile for deployment
python compile.py \
  --model-path results/mnist_large/best_model.pt \
  --benchmark-original \
  --output results/mnist_large/compiled/
```

## Benefits of Separate Scripts

- **Modularity**: Train once, evaluate many times with different settings
- **Resource efficiency**: GPU for training, CPU for compilation
- **Faster iteration**: Test evaluation or compilation without re-training
- **Cleaner code**: Each script has a single, focused responsibility

## Legacy

The original monolithic script is preserved as `main_legacy.py`.