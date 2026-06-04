#!/bin/bash
echo "$HOSTNAME"
# Set environment variables
rm -rf /tmp/huggingface_cache
mkdir -p /tmp/huggingface_cache
export HF_HOME="/tmp/huggingface_home"

export MPLCONFIGDIR="/tmp/matplotlib_cache"
export WANDB_API_KEY="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"
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


# Define the output dir inside the container
WORKDIR="$(pwd)/ALL_SCRIPTS"


# Save the original submission dir to copy files back
SUBMIT_DIR="$(pwd)"  # This is where the job was submitted from

# Load conda and activate env
source /opt/conda/etc/profile.d/conda.sh
conda activate dyna1
# Install only if missing
python -c "import transformers" 2>/dev/null || pip install transformers
python -c "import hierarchicalsoftmax" 2>/dev/null || pip install hierarchicalsoftmax
python -c "import sklearn" 2>/dev/null || pip install scikit-learn
python -c "import wandb" 2>/dev/null || pip install wandb
python -c "import yaml" 2>/dev/null || pip install pyyaml

# Move into your working code directory
cd "$WORKDIR" || exit 1




DATA_DIR="$SUBMIT_DIR/newdataset_pickles_terrsys/"

OUTPUT_DIR2="/staging/kkumari/terrsystem/longformer_hierar_wprtrn/viz"
mkdir -p "$OUTPUT_DIR2"




DATASET_NAME="repbase"


OUTPUT_DIR="/staging/kkumari/terrsystem/longformer_hierar_wprtrn/$DATASET_NAME/"

run_name="${DATASET_NAME}_longformer_hierar_Jun3"
fold=0
TRAIN_FILE=$DATA_DIR/${DATASET_NAME}/fold_${fold}_train_seqlabels.pkl
TEST_FILE=$DATA_DIR/${DATASET_NAME}/fold_${fold}_test_seqlabels.pkl
python visualize_umap.py \
  --config "$WORKDIR"/config/longformer_config.yml \
  --train_file $TRAIN_FILE \
  --test_file   $TEST_FILE \
  --model_dir "$OUTPUT_DIR"/longformer_fold${fold}_${DATASET_NAME}_longformerHierarprtrn_${fold}.pt \
  --save_dir "$OUTPUT_DIR2" \
  --run_name $run_name \
  --DPI 1000






DATASET_NAME="repetdb"

OUTPUT_DIR="/staging/kkumari/terrsystem/longformer_hierar_wprtrn/$DATASET_NAME/"

run_name="${DATASET_NAME}_longformer_hierar_Jun3"
fold=0
TRAIN_FILE=$DATA_DIR/${DATASET_NAME}/fold_${fold}_train_seqlabels.pkl
TEST_FILE=$DATA_DIR/${DATASET_NAME}/fold_${fold}_test_seqlabels.pkl
python visualize_umap.py \
  --config "$WORKDIR"/config/longformer_config.yml \
  --train_file $TRAIN_FILE \
  --test_file   $TEST_FILE \
  --model_dir "$OUTPUT_DIR"/longformer_fold${fold}_${DATASET_NAME}_longformerHierarprtrn_${fold}.pt \
  --save_dir "$OUTPUT_DIR2" \
  --run_name $run_name \
  --DPI 1000








DATASET_NAME="mntedb"

OUTPUT_DIR="/staging/kkumari/terrsystem/longformer_hierar_wprtrn/$DATASET_NAME/"

run_name="${DATASET_NAME}_longformer_hierar_Jun3"
fold=0
TRAIN_FILE=$DATA_DIR/${DATASET_NAME}/fold_${fold}_train_seqlabels.pkl
TEST_FILE=$DATA_DIR/${DATASET_NAME}/fold_${fold}_test_seqlabels.pkl
python visualize_umap.py \
  --config "$WORKDIR"/config/longformer_config.yml \
  --train_file $TRAIN_FILE \
  --test_file   $TEST_FILE \
  --model_dir "$OUTPUT_DIR"/longformer_fold${fold}_${DATASET_NAME}_longformerHierarprtrn_${fold}.pt \
  --save_dir "$OUTPUT_DIR2" \
  --run_name $run_name \
  --DPI 1000



