# ZZFormer concatenation model

This folder implements the ZZFormer **concatenation-based training variants**, where representations from transformer components and/or feature pathways are combined before prediction.

The code is organized as:
- `model/` for model definitions and fusion/concatenation architectures,
- `data/` for dataset + dataloader preparation,
- `train_*.py` / retrain scripts for experiment entry points,
- `utils.py` for metric computation and shared utilities.

---

## Folder Structure

```text
concatenation_model/
├── CHTC_scripts/
├── config/
├── data/
├── model/
├── train_zzformer_concat.py
├── retrain_onlytransformer.py
├── utils.py
├── visualize_tsne.py
├── visualize_umap.py
└── visualize_umap_heirarchical.py
```

---

## Training Scripts

### 1) `train_zzformer_concat.py`
**Purpose:** Main training entry point for the concatenation-based ZZFormer model.

**Model**  
- A concatenation/fusion model saved in the  `model/` directory.
- The script constructs the model and fuses topology and sequence information by concatenating representations from the longformer encoder and the topology encoder before classification.

**Data used:**  
- The `data/` folder contains the dataloader_cnn.py which has the TopoDataset and load_pi_lookups. They return tokenized/encoded inputs and labels in the shape expected by the concatenation model’s forward pass.

**Run (template):**
```bash
python train_zzformer_concat.py \
  --config <path-to-config> \
  --pi_dir <path to directory which has the persistence images as .tar.gz files> \
  --pretrained_mlm <checkpoint> \
  --fold <int> \
  --train_dir <path to train_pickle> \
  --val_dir <path to val_pickle> \
  --save_dir <path to save directory> \
  --run_name <name of the run> \
  --seed <random seed> \
  --wandb_project <optional wandb project name> \
  --wandb_team <optional wandb team name> \
  --wandb_dir <optional wandb directory>
```


- training hyperparameters from script (batch size, lr, epochs, seed, number of heads, number of encoder layers, feedforward dimension, GLOBAL tokens (used - BOS only), save directory, train input .pickle paths, etc.)

---



## Model Package (`model/`)

`model/` contains the architecture code used by the above training scripts.  
- Longformer encoder setup,
- feature projection layers,
- concatenation/fusion blocks,
- Hierarchical classification head,


---

## Metrics & Utilities (`utils.py`)

`utils.py` includes shared experiment utilities and metric computation, typically:
- Macro metric calculation for  precision, recall, F1, and accuracy
- 
---

## Visualization Utilities

- `visualize_umap.py`: UMAP visualization pipeline.
```bash
python visualize_umap.py \
  --config <path-to-config> \
  --train_file  <path to train_pickle> \
  --test_file  <path to val_pickle> \
  --model_dir <path to a specific fold's saved model weights .pt> \
  --save_dir <path to save directory for output embedding .npy and the visual .png> \
  --pi_dir <path to directory which has the persistence images as .tar.gz files> \
  --run_name <name of the run> \
  --DPI <dpi value for image resolution>
```



---

## Typical Workflow

1. Pretrain the Longformer encoder model from Longformer_MLM_pretraining folder alongside this folder.
2. Configure model dimensions in `config/`, make sure to keep it consistent to the pretrained weights you will be loading.
3. Prepare stratified data splits into k-folds. Prepare persistence images where the each file has images in a dictionary format - {seq: {{k}mer:persistence_image}} for k in [4, 8, 14, 20]
4. Train concat model via `train_zzformer_concat.py` with the arguments given above
5. Visualize embedding structure using the UMAP script.

---
