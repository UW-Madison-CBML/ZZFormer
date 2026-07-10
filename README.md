# ZZFormer

ZFormer provides a general framework for incorporating zigzag topological structure into sequence models, enabling learning across a wide range of biological sequence analysis problems involving tandem repeats, regulatory motifs, RNA structures, and other recurrent or cyclic patterns. 

# Usage
### Clone repo
```
git clone https://github.com/UW-Madison-CBML/ZZFormer.git
```

### Environment

Use requirements.txt to build environment or pull docker image `lsvaren/transformers`

```


```

### Zigzag homology computation

Prepare all sequences in a text file, with one sequence per line. Run `preML.py` passing a string of alphabets, memory, k-mer size, and input text file name. The output will be a `pkl` file `results_{input_file}_{k}mer.pkl`

```
cd zigzag_homology

python preML.py "A,T,G,C,X" 100 4 "dna_preML.txt"

# Output file: results_dna_preML_4mer.pkl
```

### Persistence image computation
```

```



### Train ZZFormer
Environment for ZZFormer can be loaded using the Docker image -  'kritikakumari22/tda_seqemb:zzformer_transformeronly5'
In directory ./ZZFormer/concatenation_model/ one can do k-fold cross validation using-
```
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
```


# Citation
```
@article{ZZFormer,
  author=Kritika Kumari and Levi Svaren1, Dhananjay Bhaskar,
  title=ZZFORMER: A SLIDING WINDOW ZIGZAG PERSISTENT
HOMOLOGY TRANSFORMER FOR REPETITIVE SEQUENCES
}
```




