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








# DATASET_NAME="mntedb"
# DATASET_NAME2="MnTEdb"


# DATA_DIR="/staging/groups/bhaskar_group/zzformer_topo/"
# IMG_DIR="/staging/groups/bhaskar_group/terrier_zzformer/"
# OUTPUT_DIR="/staging/k/kkumari/terrsystem/ZZFormer_Concat/$DATASET_NAME/"
# OUTPUT_DIR2="/staging/k/kkumari/terrsystem/ZZFormer_Concat/viz"
# mkdir -p "$OUTPUT_DIR2"

# fold=0
# run_name="${DATASET_NAME2}_ZZConcat_${fold}"
# TRAIN_FILE=$DATA_DIR/${DATASET_NAME2}_zzformer_train_fold_${fold}.pkl
# TEST_FILE=$DATA_DIR/${DATASET_NAME2}_zzformer_test_fold_${fold}.pkl
# IMG_FILE=$IMG_DIR/${DATASET_NAME2}/

# python visualize_umap.py \
#   --config "$WORKDIR"/config/longformer_config.yml \
#   --train_file $TRAIN_FILE \
#   --test_file   $TEST_FILE \
#   --model_dir "$OUTPUT_DIR"/${DATASET_NAME}_ZZConcat_${fold}.pt \
#   --save_dir "$OUTPUT_DIR2" \
#   --pi_dir $IMG_FILE \
#   --run_name $run_name \
#   --DPI 1000









DATASET_NAME="repbase"
DATASET_NAME2="Repbase"


DATA_DIR="/staging/groups/bhaskar_group/zzformer_topo/"
IMG_DIR="/staging/groups/bhaskar_group/terrier_zzformer/"
OUTPUT_DIR="/staging/k/kkumari/terrsystem/ZZFormer_Concat/$DATASET_NAME/"
OUTPUT_DIR2="/staging/k/kkumari/terrsystem/ZZFormer_Concat/viz"
mkdir -p "$OUTPUT_DIR2"

fold=0
run_name="${DATASET_NAME2}_ZZConcat_${fold}"
TRAIN_FILE=$DATA_DIR/${DATASET_NAME2}_zzformer_train_fold_${fold}.pkl
TEST_FILE=$DATA_DIR/${DATASET_NAME2}_zzformer_test_fold_${fold}.pkl
IMG_FILE=$IMG_DIR/${DATASET_NAME2}/

python visualize_umap.py \
  --config "$WORKDIR"/config/longformer_config.yml \
  --train_file $TRAIN_FILE \
  --test_file   $TEST_FILE \
  --model_dir "$OUTPUT_DIR"/${DATASET_NAME}_ZZConcat_${fold}.pt \
  --save_dir "$OUTPUT_DIR2" \
  --pi_dir $IMG_FILE \
  --run_name $run_name \
  --DPI 1000


























DATASET_NAME="repetdb"
DATASET_NAME2="RepetDB"


DATA_DIR="/staging/groups/bhaskar_group/zzformer_topo/"
IMG_DIR="/staging/groups/bhaskar_group/terrier_zzformer/"
OUTPUT_DIR="/staging/k/kkumari/terrsystem/ZZFormer_Concat/$DATASET_NAME/"
OUTPUT_DIR2="/staging/k/kkumari/terrsystem/ZZFormer_Concat/viz"
mkdir -p "$OUTPUT_DIR2"

fold=0
run_name="${DATASET_NAME2}_ZZConcat_${fold}"
TRAIN_FILE=$DATA_DIR/${DATASET_NAME2}_zzformer_train_fold_${fold}.pkl
TEST_FILE=$DATA_DIR/${DATASET_NAME2}_zzformer_test_fold_${fold}.pkl
IMG_FILE=$IMG_DIR/${DATASET_NAME2}/

python visualize_umap.py \
  --config "$WORKDIR"/config/longformer_config.yml \
  --train_file $TRAIN_FILE \
  --test_file   $TEST_FILE \
  --model_dir "$OUTPUT_DIR"/${DATASET_NAME}_ZZConcat_${fold}.pt \
  --save_dir "$OUTPUT_DIR2" \
  --pi_dir $IMG_FILE \
  --run_name $run_name \
  --DPI 1000











