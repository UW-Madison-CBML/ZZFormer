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
  --fold <int> \
  --device <cuda|cpu>
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


These scripts are useful for post-training representation analysis and class separability checks.

---

## Typical Workflow

1. Configure experiment in `config/`.
2. Train concat model via `train_zzformer_concat.py`.
3. Optionally run transformer-only retraining with `retrain_onlytransformer.py`.
4. Evaluate metrics (from training logs / utility routines).
5. Visualize embedding structure using TSNE/UMAP scripts.

---

## Notes

- Ensure feature dimensions in config align with the concatenation layer expectations.
- Keep tokenizer/preprocessing settings synchronized across training and retraining scripts.
