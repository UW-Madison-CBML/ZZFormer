#!/bin/bash

echo $HOSTNAME

pip install torch==2.10.0
pip install hierarchicalsoftmax

ls *

label_path="./MnTEdb/hsm_MnTEdb_labels.tsv"
pi_dir='MnTEdb/'
save_name="MnTEdb"


#python train_hier.py $label_path $pi_dir $save_name
python train_new.py $label_path $pi_dir $save_name

#mkdir mntedb_results
#mv *.pkl *joblib *pth *npz mntedb_results
#tar -czvf $save_name.tar.gz mntedb_results

echo DONE
