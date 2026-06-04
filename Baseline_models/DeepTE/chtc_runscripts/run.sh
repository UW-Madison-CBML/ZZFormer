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

SUBMIT_DIR="$(pwd)"  # This is where the job was submitted from

WORKDIR="$SUBMIT_DIR/DeepTE"
pip install -U "scikit-learn>=0.22"
python -c "import sklearn; print(sklearn.__version__)"


cd "$WORKDIR" || exit 1


BASE_DIR="$WORKDIR"



# DATASET_NAMES=("mntedb")
DATASET_NAMES=("mntedb" "repetdb" "repbase")

for DATASET_NAME in "${DATASET_NAMES[@]}"; do
  echo "=== Dataset $DATASET_NAME ==="
  
  
  data_dir="${BASE_DIR}/Data/DeepTE/${DATASET_NAME}"

  # DATASET_NAME="repbase"
  for fold in {0..4}; do
    echo "=== Fold $fold ==="
    INPUT_FASTA_TRAIN="${data_dir}/fold_${fold}_train_${DATASET_NAME}.txt"
    INPUT_FASTA_TEST="${data_dir}/fold_${fold}_test_${DATASET_NAME}.txt"


    OUT_DIR="/staging/kkumari/DEEPTE/${DATASET_NAME}/${fold}/order/"
    mkdir -p "${OUT_DIR}"

    python train_deepte.py $INPUT_FASTA_TRAIN $INPUT_FASTA_TEST 0 $OUT_DIR "${DATASET_NAME}_order_${fold}"


    OUT_DIR="/staging/kkumari/DEEPTE/${DATASET_NAME}/${fold}/SF/"
    mkdir -p "${OUT_DIR}"

    python train_deepte.py $INPUT_FASTA_TRAIN $INPUT_FASTA_TEST 1 $OUT_DIR "${DATASET_NAME}_superfamily_${fold}"


  done
done







