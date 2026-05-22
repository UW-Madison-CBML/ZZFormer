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
WORKDIR="/app/concat_MLP"

# Save the original submission dir to copy files back
SUBMIT_DIR="$(pwd)"  # This is where the job was submitted from

# Load conda and activate env
source /opt/conda/etc/profile.d/conda.sh
conda activate dyna1

# Move into your working code directory
cd "$WORKDIR" || exit 1


DATASET_NAME="mntedb"

PRETRAINED_MODEL="/staging/kkumari/pretrainbruns2/mlm_best.pt"


# Loop over folds 0 to 4
for fold in {0..4}; do
  run_name="Order_${DATASET_NAME}_Transformeronly_fold${fold}"
  TRAIN_FILE="$WORKDIR/data/pickles/${DATASET_NAME}/fold_${fold}_train_seqlabels.pkl"
  TEST_FILE="$WORKDIR/data/pickles/${DATASET_NAME}/fold_${fold}_test_seqlabels.pkl"
  OUTPUT_DIR="/staging/kkumari/zzformer_transformeronly/$DATASET_NAME/order/"

  mkdir -p "$OUTPUT_DIR"

  python retrain_onlytransformer.py \
  --config "$WORKDIR"/config/ffn_config_transformeronly.yml \
  --mode classify_order \
  --fold $fold \
  --pretrained_mlm $PRETRAINED_MODEL \
  --train_dir $TRAIN_FILE \
  --val_dir   $TEST_FILE \
  --save_dir $OUTPUT_DIR \
  --wandb_project zzformer \
  --wandb_team 'kkumari-university-of-wisconsin-madison'  \
  --wandb_dir "/tmp/wandb" \
  --run_name  $run_name\
  --seed 22


done

# Loop over folds 0 to 4
for fold in {0..4}; do
  run_name="SF_${DATASET_NAME}_Transformeronly_fold${fold}"
  TRAIN_FILE="$WORKDIR/data/pickles/${DATASET_NAME}/fold_${fold}_train_seqlabels.pkl"
  TEST_FILE="$WORKDIR/data/pickles/${DATASET_NAME}/fold_${fold}_test_seqlabels.pkl"
  OUTPUT_DIR="/staging/kkumari/zzformer_transformeronly/$DATASET_NAME/SF/"

  mkdir -p "$OUTPUT_DIR"

  python retrain_onlytransformer.py \
  --config "$WORKDIR"/config/ffn_config_transformeronly.yml \
  --mode classify_sf \
  --fold $fold \
  --pretrained_mlm $PRETRAINED_MODEL \
  --train_dir $TRAIN_FILE \
  --val_dir   $TEST_FILE \
  --save_dir $OUTPUT_DIR \
  --wandb_project zzformer \
  --wandb_team 'kkumari-university-of-wisconsin-madison'  \
  --wandb_dir "/tmp/wandb" \
  --run_name  $run_name\
  --seed 22


done





