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
WORKDIR="/app/Pretraining_Baselines"

# Save the original submission dir to copy files back
SUBMIT_DIR="$(pwd)/ALL_SCRIPTS" # This is where the job was submitted from
raw_submit_dir="$(pwd)/wholes/" # This is the original submission directory
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
cd "$SUBMIT_DIR" || exit 1

run_name="pretraining_reponly_May28"
OUTPUT_DIR="/staging/kkumari/terrsystem/pretraining_longformerreponlyMay28/"
mkdir -p "$OUTPUT_DIR"







OUTPUT_DIR="/staging/kkumari/terrsystem/pretraining_longformerreponlyMay28/"

OUTPUT_DIR2="/staging/kkumari/terrsystem/pretraining_longformerreponlyMay28/viz"
mkdir -p "$OUTPUT_DIR2"




DATASET_NAME="repbase"

run_name="${DATASET_NAME}_reponlyvis_longformer_May28"

python "$SUBMIT_DIR"/visualize_umap.py \
  --config "$SUBMIT_DIR"/config/longformer_mlm_config.yml \
  --seq_file "$raw_submit_dir"/${DATASET_NAME}_whole.pkl \
  --model_dir "$OUTPUT_DIR"/longformer_mlm_pretraining_reponly_May28.pt \
  --save_dir "$OUTPUT_DIR2" \
  --run_name $run_name \
  --DPI 1000






DATASET_NAME="repetdb"

run_name="${DATASET_NAME}_reponlyvis_longformer_May28"

python "$SUBMIT_DIR"/visualize_umap.py \
  --config "$SUBMIT_DIR"/config/longformer_mlm_config.yml \
  --seq_file "$raw_submit_dir"/${DATASET_NAME}_whole.pkl \
  --model_dir "$OUTPUT_DIR"/longformer_mlm_pretraining_reponly_May28.pt \
  --save_dir "$OUTPUT_DIR2" \
  --run_name "$run_name" \
  --DPI 1000







DATASET_NAME="mntedb"

run_name="${DATASET_NAME}_reponlyvis_longformer_May28"

python "$SUBMIT_DIR"/visualize_umap.py \
  --config "$SUBMIT_DIR"/config/longformer_mlm_config.yml \
  --seq_file "$raw_submit_dir"/${DATASET_NAME}_whole.pkl \
  --model_dir "$OUTPUT_DIR"/longformer_mlm_pretraining_reponly_May28.pt \
  --save_dir "$OUTPUT_DIR2" \
  --run_name $run_name \
  --DPI 1000

