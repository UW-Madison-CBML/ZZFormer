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

# Load conda and activate env
source /opt/conda/etc/profile.d/conda.sh
conda activate dyna1

# Move into your working code directory
cd "$SUBMIT_DIR" || exit 1

run_name="pretraining_Mar6"
OUTPUT_DIR="/staging/kkumari/Final_transformerMar6/pretraining_all3dbscomb"
mkdir -p "$OUTPUT_DIR"

# Run your Python script
python train.py \
  --config "$SUBMIT_DIR"/config/main_cng.yml \
  --mode mlm \
  --train_dir "$WORKDIR"/ALL_SEQ_Unique.pkl \
  --save_dir "$OUTPUT_DIR" \
  --wandb_project zzformer \
  --wandb_team 'kkumari-university-of-wisconsin-madison'  \
  --wandb_dir "/tmp/wandb" \
  --run_name  $run_name\
  --seed 22












OUTPUT_DIR2="/staging/kkumari/Final_transformerMar6/pretraining_all3dbscomb/viz"
mkdir -p "$OUTPUT_DIR2"

DATASET_NAME="repbase"

run_name="${DATASET_NAME}_vis_w_prtrnd1024len"

python "$WORKDIR"/visualize_umap.py \
  --config "$SUBMIT_DIR"/config/main_cng.yml \
  --seq_file "$WORKDIR"/data/all_dbs_seqs_andlabels/${DATASET_NAME}_all_sequences_labels.pkl \
  --model_dir "$OUTPUT_DIR"/mlm_best.pt \
  --save_dir "$OUTPUT_DIR2" \
  --run_name $run_name \
  --DPI 1000






DATASET_NAME="repetdb"

run_name="${DATASET_NAME}_vis_w_prtrnd1024len"

python "$WORKDIR"/visualize_umap.py \
  --config "$SUBMIT_DIR"/config/main_cng.yml \
  --seq_file "$WORKDIR"/data/all_dbs_seqs_andlabels/${DATASET_NAME}_all_sequences_labels.pkl \
  --model_dir "$OUTPUT_DIR"/mlm_best.pt \
  --save_dir "$OUTPUT_DIR2" \
  --run_name $run_name \
  --DPI 1000







DATASET_NAME="mntedb"

run_name="${DATASET_NAME}_vis_w_prtrnd1024len"

python "$WORKDIR"/visualize_umap.py \
  --config "$SUBMIT_DIR"/config/main_cng.yml \
  --seq_file "$WORKDIR"/data/all_dbs_seqs_andlabels/${DATASET_NAME}_all_sequences_labels.pkl \
  --model_dir "$OUTPUT_DIR"/mlm_best.pt \
  --save_dir "$OUTPUT_DIR2" \
  --run_name $run_name \
  --DPI 1000

