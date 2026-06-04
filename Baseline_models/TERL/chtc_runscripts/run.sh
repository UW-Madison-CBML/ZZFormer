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
DATASET_NAMES=("repbase" "mntedb" "repetdb")

for DATASET_NAME in "${DATASET_NAMES[@]}"; do
  echo "=== Dataset $DATASET_NAME ==="
  
  
  data_dir="${BASE_DIR}/Data/TERL/"

  # DATASET_NAME="repbase"
  for fold in {0..4}; do
    echo "=== Fold $fold ==="


    INPUT_TRAIN_SF="${data_dir}/${DATASET_NAME}/fold_${fold}/Superfamily/"
    OUT_DIR_SF="/staging/kkumari/terrsystem/TERL/${DATASET_NAME}/SF/${fold}"
    mkdir -p "${OUT_DIR_SF}"

    MODEL_NAME="fold${fold}_${DATASET_NAME}_sf"

    python3 terl_train.py -r $INPUT_TRAIN_SF -p ${MODEL_NAME} -md $OUT_DIR_SF  -od $OUT_DIR_SF -sm -sr -sg



    INPUT_TRAIN_ORDER="${data_dir}/${DATASET_NAME}/fold_${fold}/Order/"
    OUT_DIR_ORDER="/staging/kkumari/terrsystem/TERL/${DATASET_NAME}/order/${fold}"
    mkdir -p "${OUT_DIR_ORDER}"


    MODEL_NAME="fold${fold}_${DATASET_NAME}_order"
    python3 terl_train.py -r $INPUT_TRAIN_ORDER -p ${MODEL_NAME} -md $OUT_DIR_ORDER -od $OUT_DIR_ORDER -sm -sr -sg


  done
done














# DATASET_NAME="repbase"


python summarize_fold_terl.py -s /staging/kkumari/terrsystem/TERL -d mntedb   -m TERL -o /staging/kkumari/terrsystem/TERL/cross5val_mntedb_summary.csv
python summarize_fold_terl.py -s /staging/kkumari/terrsystem/TERL -d repetdb  -m TERL -o /staging/kkumari/terrsystem/TERL/cross5val_repetdb_summary.csv
python summarize_fold_terl.py -s /staging/kkumari/terrsystem/TERL -d repbase  -m TERL -o /staging/kkumari/terrsystem/TERL/cross5val_repbase_summary.csv

