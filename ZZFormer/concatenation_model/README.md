# ZZFormerconcatenation model

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

**Arguments (to be finalized from script parser):**
- `--config`
- `--fold`
- `--device`
- training hyperparameters from script (batch size, lr, epochs, seed, output/checkpoint paths, etc.)

---

### 2) `retrain_onlytransformer.py`
**Purpose:** Retraining/fine-tuning workflow focused on the transformer-only branch (without full concatenation pathway, or with selective branch freezing depending on script logic).

**Model called:**  
- Transformer-focused model path, usually from `model/` modules or direct transformer class instantiation.

**What this model is:**  
- A reduced/specialized variant used to isolate transformer contribution or perform staged retraining.

**Data pipeline used:**  
- Reuses `data/` dataloaders with potentially different config settings for retraining experiments.

**Run (template):**
```bash
python retrain_onlytransformer.py \
  --config <path-to-config> \
  --fold <int> \
  --device <cuda|cpu> \
  --checkpoint <optional-initial-weights>
```

**Arguments (to be finalized from script parser):**
- `--config`, `--fold`, `--device`
- optional checkpoint/pretrained args
- optimizer/scheduler and runtime args as defined in script

---

## Model Package (`model/`)

`model/` contains the architecture code used by the above training scripts.  
Typical responsibilities:
- Transformer encoder setup,
- feature projection layers,
- concatenation/fusion blocks,
- classification heads,
- forward variants for full concat vs transformer-only retraining.

---

## Data Package (`data/`)

`data/` provides:
- dataset wrappers,
- fold split handling,
- tokenization/encoding,
- collate functions matching expected model inputs for fusion/retraining modes.

---

## Metrics & Utilities (`utils.py`)

`utils.py` includes shared experiment utilities and metric computation, typically:
- metric calculation for classification performance,
- logging helpers,
- seed/setup helpers,
- checkpoint and result formatting utilities.

---

## Visualization Utilities

- `visualize_tsne.py`: t-SNE projection for learned embeddings/features.
- `visualize_umap.py`: UMAP visualization pipeline.
- `visualize_umap_heirarchical.py`: hierarchical UMAP analysis variant.

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

- Use consistent fold definitions and seeds when comparing concat vs transformer-only runs.
- Ensure feature dimensions in config align with the concatenation layer expectations.
- Keep tokenizer/preprocessing settings synchronized across training and retraining scripts.
