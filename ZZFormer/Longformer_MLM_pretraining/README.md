# Longformer_MLM_pretraining

This folder contains the Longformer masked language model (MLM) pretraining workflow and UMAP-based embedding visualization scripts used in the ZZFormer pipeline.

The project is structured with:
- model-training entrypoint: `pretrain_longformer_mlm.py`
- visualization entrypoint: `visualize_umap.py`
- data loading/preprocessing logic in `data/`
- experiment configs in `config/`
- cluster execution scripts in `CHTC_scripts/`

---

## Folder Structure

```text
Longformer_MLM_pretraining/
├── CHTC_scripts/
│   └── train.sh
├── config/
│   └── longformer_mlm_config.yml
├── data/
├── pretrain_longformer_mlm.py
├── utils.py
└── visualize_umap.py
```

---

## Training Script

### `pretrain_longformer_mlm.py`
  
Pretrains a Longformer encoder using masked language modeling so the learned weights can be reused in downstream ZZFormer training workflows. We use LongformerForMaskedLM to perform this pretraining; these weights are transferable to any Longformer encoder later.

**model:**  
- Longformer-based MLM model for long-context sequence modeling.
- Breaks down attention into local and global attention. Has the transformer encoder + linear head for masked-token prediction over the token vocab -
    ''{
      "PAD":  0,
      "a":    1, "c": 2, "g": 3, "t": 4,
      "x":    5,
      "BOS":  6,
      "EOS":  7,
      "MASK": 8,
  }''
- Produces pretrained checkpoint(s), including `mlm_best.pt`, which is later consumed by visualization and downstream task scripts.
- Training integrates experiment tracking via Weights & Biases (W&B) using explicit project/team/run settings in the CHTC script.

**Input Data:**  
- Training input is passed as a pickled sequence dataset via:
  - `--train_dir "$raw_submit_dir"/RepBase31pt04.pkl`
- Data preparation and batching are handled by code under `data/`, aligned with the tokenizer/config specified in `config/longformer_mlm_config.yml`.

**Run command used in `CHTC_scripts/train.sh`:**
```bash
python pretrain_longformer_mlm.py \
  --config "$SUBMIT_DIR"/config/longformer_mlm_config.yml \
  --train_dir "$raw_submit_dir"/RepBase31pt04_converted2_terrierlabelsMay13.pkl \
  --save_dir "$OUTPUT_DIR" \
  --wandb_project ZZFORMER_Terrierlabeling \
  --wandb_team 'kkumari-university-of-wisconsin-madison'  \
  --wandb_dir "/tmp/wandb" \
  --run_name  "$run_name" \
  --seed 22
```

---

## Visualization Script

### `visualize_umap.py`

**Purpose:**  
Loads pretrained MLM weights and visualizes learned sequence representations using UMAP.

**model:**  
- Reuses the pretrained Longformer weights from:
  - `--model_dir "$OUTPUT_DIR"/mlm_best.pt`
- Generates embedding projections for different datasets, enabling qualitative comparison of representation structure across corpora.

**Input Data:**  
The CHTC script runs visualization on three sequence datasets via `--seq_file`:
1. `RepBase31pt04_converted2_terrierlabelsMay13.pkl`
2. `Repetdb_allfiltered_terrier_May13.pkl`
3. `MnTEdb_terrier.pkl`

All are read from `$raw_submit_dir`.

**Run commands used in `CHTC_scripts/train.sh`:**

**Repbase**
```bash
python "$SUBMIT_DIR"/visualize_umap.py \
  --config "$SUBMIT_DIR"/config/longformer_mlm_config.yml \
  --seq_file "$raw_submit_dir"/RepBase31pt04_converted2_terrierlabelsMay13.pkl \
  --model_dir "$OUTPUT_DIR"/mlm_best.pt \
  --save_dir "$OUTPUT_DIR2" \
  --run_name $run_name \
  --DPI 1000
```

**RepetDB**
```bash
python "$SUBMIT_DIR"/visualize_umap.py \
  --config "$SUBMIT_DIR"/config/longformer_mlm_config.yml \
  --seq_file "$raw_submit_dir"/Repetdb_allfiltered_terrier_May13.pkl \
  --model_dir "$OUTPUT_DIR"/mlm_best.pt \
  --save_dir "$OUTPUT_DIR2" \
  --run_name "$run_name" \
  --DPI 1000
```

**MnTEdb**
```bash
python "$SUBMIT_DIR"/visualize_umap.py \
  --config "$SUBMIT_DIR"/config/longformer_mlm_config.yml \
  --seq_file "$raw_submit_dir"/MnTEdb_terrier.pkl \
  --model_dir "$OUTPUT_DIR"/mlm_best.pt \
  --save_dir "$OUTPUT_DIR2" \
  --run_name $run_name \
  --DPI 1000
```


---

## Config and CHTC Execution Notes

- Config file used by both training and visualization:
  - `config/longformer_mlm_config.yml`
- CHTC launcher script:
  - `CHTC_scripts/train.sh`
- Key runtime outputs in script:
  - Training checkpoints: `"$OUTPUT_DIR"`
  - UMAP outputs: `"$OUTPUT_DIR2"` (set to `"$OUTPUT_DIR"/viz`)

---

## Typical Workflow

1. Set up environment and paths (as done in `train.sh`).
2. Run MLM pretraining with `pretrain_longformer_mlm.py`.
3. Use generated `mlm_best.pt` for UMAP visualization with `visualize_umap.py`.
4. Compare final representations across Repbase, RepetDB, and MnTEdb outputs using the visualizations.

---

