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
GROUP_STAGING="/staging/groups/bhaskar_group/seq_embedding/"


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

DATASET_NAME="repetdb"



PRETRAINED_MODEL="/staging/kkumari/terrsystem/pretraining_longformerreponlyMay28/longformer_mlm_pretraining_reponly_May28.pt"


DATA_DIR="$SUBMIT_DIR/newdataset_pickles_terrsys/"
OUTPUT_DIR="/staging/kkumari/terrsystem/longformer_runs_wprtrn/$DATASET_NAME/"
mkdir -p "$OUTPUT_DIR"

# Loop over folds 0 to 4
for fold in {0..4}; do
  run_name="${DATASET_NAME}_longformerpretrn_${fold}"
  TRAIN_FILE=$DATA_DIR/${DATASET_NAME}/fold_${fold}_train_seqlabels.pkl
  TEST_FILE=$DATA_DIR/${DATASET_NAME}/fold_${fold}_test_seqlabels.pkl
  
  python train_longformer_withpretrnwts_nothierarchical.py \
  --config "$WORKDIR"/config/longformer_config.yml \
  --fold $fold \
  --pretrained_mlm $PRETRAINED_MODEL \
  --train_dir $TRAIN_FILE \
  --val_dir   $TEST_FILE \
  --save_dir $OUTPUT_DIR \
  --wandb_project ZZFORMER_Terrierlabeling \
  --wandb_team 'kkumari-university-of-wisconsin-madison'  \
  --wandb_dir "/tmp/wandb" \
  --run_name  $run_name \
  --seed 22


done
