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
GROUP_STAGING="/staging/groups/bhaskar_group/zzformer_050526/"


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
pip install hierarchicalsoftmax

# Move into your working code directory
cd "$WORKDIR" || exit 1

DATASET_NAME_2="MnTEdb"

DATASET_NAME="mntedb"

PRETRAINED_MODEL="/staging/kkumari/Final_transformerMar11/pretrained_wts/mlm_best.pt"


# fold=0
# Loop over folds 0 to 4
for fold in {0..4}; do
  run_name="HierrchlCA_${DATASET_NAME}_fold${fold}"
  OUTPUT_DIR="/staging/kkumari/ZZFORMER_May22/$DATASET_NAME/"


  mkdir -p "$OUTPUT_DIR"

  python train_withCrossAten.py \
    --config "${WORKDIR}/config/ffn_config_transformeronly.yml" \
    --fold "${fold}" \
    --pretrained_mlm "${PRETRAINED_MODEL}" \
    --labels_path "${GROUP_STAGING}/${DATASET_NAME_2}/hsm_${DATASET_NAME_2}_labels.tsv" \
    --mer4_dir "${GROUP_STAGING}/${DATASET_NAME_2}/4mer" \
    --mer8_dir "${GROUP_STAGING}/${DATASET_NAME_2}/8mer" \
    --mer14_dir "${GROUP_STAGING}/${DATASET_NAME_2}/14mer" \
    --mer20_dir "${GROUP_STAGING}/${DATASET_NAME_2}/20mer" \
    --save_dir "${OUTPUT_DIR}" \
    --wandb_project "ZZFORMER_April26" \
    --wandb_team 'kkumari-university-of-wisconsin-madison'  \
    --wandb_dir "/tmp/wandb" \
    --run_name  "${run_name}"\
    --seed 22


done




# condor_submit mntedb.sub
# condor_submit mntedb_sf.sub
# condor_submit repetdb.sub
# condor_submit repetdb_sf.sub
# condor_submit repbase.sub
# condor_submit repbase_sf.sub