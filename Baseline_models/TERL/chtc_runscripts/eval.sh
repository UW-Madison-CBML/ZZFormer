#!/bin/bash
export USER="${USER:-root}"
export LOGNAME="${LOGNAME:-root}"
export HOME="${HOME:-/tmp}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/torch_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg_cache}"
mkdir -p "$HOME" "$TORCHINDUCTOR_CACHE_DIR" "$XDG_CACHE_HOME"




rm -rf /tmp/huggingface_cache
mkdir -p /tmp/huggingface_cache
export HF_HOME="/tmp/huggingface_home"


export MPLCONFIGDIR="/tmp/matplotlib_cache"
export WANDB_API_KEY="8b6cf4ca623540253084fe0b0f640583966540d0"
export WANDB_DIR="/tmp/wandb"
export WANDB_CONFIG_DIR="/tmp/wandb"
export WANDB_CACHE_DIR="/tmp/wandb"


mkdir -p /tmp/wandb
export WANDB_DIR="/tmp/wandb"

# Create cache directories
mkdir -p "$TRANSFORMERS_CACHE"
mkdir -p "$HF_HOME"
mkdir -p "$MPLCONFIGDIR"

mkdir -p "$WANDB_DIR" "$WANDB_CONFIG_DIR" "$WANDB_CACHE_DIR"

pip install -U "scikit-learn>=0.22"
python -c "import sklearn; print(sklearn.__version__)"

SUBMIT_DIR="$(pwd)"  # This is where the job was submitted from

WORKDIR="$SUBMIT_DIR/TERL_Code"

cd "$WORKDIR" || exit 1


BASE_DIR="$WORKDIR"



# DATASET_NAMES=("mntedb" "repetdb" "repbase")

DATASET_NAME="repbase"


python summarize_fold_terl.py -s /staging/kkumari/TERL -d mntedb   -m TERL -o /staging/kkumari/TERL/cross5val_mntedb_summary.csv
python summarize_fold_terl.py -s /staging/kkumari/TERL -d repetdb  -m TERL -o /staging/kkumari/TERL/cross5val_repetdb_summary.csv
python summarize_fold_terl.py -s /staging/kkumari/TERL -d repbase  -m TERL -o /staging/kkumari/TERL/cross5val_repbase_summary.csv

