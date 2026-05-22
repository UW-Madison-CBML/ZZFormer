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
SUBMIT_DIR="$(pwd)"  # This is where the job was submitted from

# Load conda and activate env
source /opt/conda/etc/profile.d/conda.sh
conda activate dyna1

# Move into your working code directory
cd "$WORKDIR" || exit 1





DATASET_NAME="repbase"

run_name="${DATASET_NAME}_vis_w_prtrnd1500len"

python visualize_umap.py \
  --config "$WORKDIR"/config/main_cng.yml \
  --seq_file "$WORKDIR"/data/all_dbs_seqs_andlabels/${DATASET_NAME}_all_sequences_labels.pkl \
  --model_dir /staging/kkumari/pretrainbruns2/mlm_best.pt \
  --save_dir /staging/kkumari/pretrainbruns2/ \
  --run_name $run_name \
  --DPI 1000






DATASET_NAME="repetdb"

run_name="${DATASET_NAME}_vis_w_prtrnd1500len"

python visualize_umap.py \
  --config "$WORKDIR"/config/main_cng.yml \
  --seq_file "$WORKDIR"/data/all_dbs_seqs_andlabels/${DATASET_NAME}_all_sequences_labels.pkl \
  --model_dir /staging/kkumari/pretrainbruns2/mlm_best.pt \
  --save_dir /staging/kkumari/pretrainbruns2/ \
  --run_name $run_name \
  --DPI 1000







DATASET_NAME="mntedb"

run_name="${DATASET_NAME}_vis_w_prtrnd1500len"

python visualize_umap.py \
  --config "$WORKDIR"/config/main_cng.yml \
  --seq_file "$WORKDIR"/data/all_dbs_seqs_andlabels/${DATASET_NAME}_all_sequences_labels.pkl \
  --model_dir /staging/kkumari/pretrainbruns2/mlm_best.pt \
  --save_dir /staging/kkumari/pretrainbruns2/ \
  --run_name $run_name \
  --DPI 1000

