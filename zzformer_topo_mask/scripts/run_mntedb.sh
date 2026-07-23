#!/bin/bash

echo $HOSTNAME

pip install torch==2.10.0
pip install hierarchicalsoftmax

ls *

label_path="./MnTEdb/mntedb_QC_051326.csv"
pi_dir='MnTEdb/'
save_name="MnTEdb"

WORKDIR="./"


python train_zzformer_concat.py \
        --config "$WORKDIR"/config/longformer_config.yml \
        --fold $fold \
        --pretrained_mlm $PRETRAINED_MODEL \
        --train_dir $TRAIN_FILE \
        --val_dir   $TEST_FILE \
        --save_dir $OUTPUT_DIR \
        --pi_dir $IMG_FILE \
        --wandb_project ZZFORMER_Terrierlabeling \
        --wandb_team 'kkumari-university-of-wisconsin-madison'  \
        --wandb_dir "/tmp/wandb" \
        --run_name  $run_name \
        --seed 22

python train_dim2.py $label_path $pi_dir $save_name

echo $label_path
echo $pi_dir
echo $save_name
echo DONE
