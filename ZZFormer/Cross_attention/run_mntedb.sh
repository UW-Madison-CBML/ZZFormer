#!/bin/bash

pip install hierarchicalsoftmax
pip install torch==2.10.0
pip install accelerate
pip install wandb

echo "$HOSTNAME"
# Set environment variables
rm -rf /tmp/huggingface_cache
mkdir -p /tmp/huggingface_cache
export HF_HOME="/tmp/huggingface_home"
export MPLCONFIGDIR="/tmp/matplotlib_cache"
export WANDB_API_KEY="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" # Levi key
export WANDB_DIR="/tmp/wandb"
export WANDB_CONFIG_DIR="/tmp/wandb"
export WANDB_CACHE_DIR="/tmp/wandb"
# GROUP_STAGING="/staging/groups/bhaskar_group/seq_embedding/"
mkdir -p /tmp/wandb
export WANDB_DIR="/tmp/wandb"

# Create cache directories
mkdir -p "$TRANSFORMERS_CACHE"
mkdir -p "$HF_HOME"
mkdir -p "$MPLCONFIGDIR"

mkdir -p "$WANDB_DIR" "$WANDB_CONFIG_DIR" "$WANDB_CACHE_DIR"
export PYTHONUNBUFFERED=1

# Save the original submission dir to copy files back
# SUBMIT_DIR="$(pwd)"  
# PRETRAINED_MODEL="/staging/k/kkumari/terrsystem/pretraining_longformerreponlyMay28/longformer_mlm_pretraining_reponly_May28.pt"
PRETRAINED_MODEL="/staging/groups/bhaskar_group/zzformer_hash/weights/longformer_mlm_pretraining_reponly_May28.pt"

# FOLD=0
RUN_NAME="MnTEdb_test"
DATASET_NAME="mntedb"


# DATASET_NAME2="MnTEdb"

# DATA_DIR="/staging/groups/bhaskar_group/zzformer_topo/"
IMG_DIR="/staging/s/svaren/072026/mntedb/"
OUTPUT_DIR="/staging/s/svaren/072026/cross_attention/$DATASET_NAME/"
mkdir -p "$OUTPUT_DIR"

# Run model training
for FOLD in {0..4}; do
        echo "Fold: ${FOLD}"
        python train_zzformer_CA_svaren.py \
                --config longformer_config.yml \
                --fold $FOLD \
                --run_name  $RUN_NAME \
                --pretrained_mlm $PRETRAINED_MODEL \
                --save_dir $OUTPUT_DIR \
                --pi_dir $IMG_DIR \
                --wandb_project zzformer \
                --wandb_team 'svaren-uni'  \
                --wandb_dir "/tmp/wandb" \
                --seed 22
done
        # --train_dir $TRAIN_FILE \
        # --val_dir   $TEST_FILE \

