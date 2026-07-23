# TeClass2

This folder contains the Longformer-based TeClass2 training workflow as a baseline comparison for our hierarchical sequence classification ZZFormer pipeline.

The project is structured with:
- model training and data loading/preprocessing entry point: `train_longformer.py`
- TeClass-2-based configuration setup in `config/`

---

## Folder Structure

```text
TeClass2/
├── config/
├── data/
├── model/
├── train_longformer.py
└── utils.py
```

---

## Training Script

### `train_longformer.py`

Trains a Longformer-based TeClass2 classifier for sequence-level prediction tasks. The script orchestrates data loading, model construction, optimization, evaluation, and checkpoint tracking across experiments.

**model:**  
- Longformer-based architecture for long-context sequence classification.
- Uses sparse attention (local + global attention behavior from Longformer) to process long tokenized biological sequences efficiently.
- The classification head is attached to the encoder output to perform supervised prediction over target classes.
- In this folder, the training script calls model definitions from `model/` (and associated transformer initialization logic), following the TeClass2 architecture setup used for downstream labeling/classification.

**Input Data:**  
- Format: a pickle file containing a dictionary in the format {sequence:(order/superfamily)} where sequence alphabets are all in small caps.
- Input is loaded through torch dataloaders; the user must provide their own train and validation data split file paths.
- Data preprocessing includes tokenized sequence encoding aligned to the Longformer configuration used by this folder.
- Batches are structured for supervised classification (input tensors + label tensors), and fold-based or split-based training is handled by the CHTC scripts pipeline.

**Run:**
To run locally, use the Docker image: kritikakumari22/tda_seqemb:zzformer_transformeronly5
Run from the same directory in which train_longformer.py exists.
```bash
python train_longformer.py \
  --config <path-to-config> \
  --train_dir <path-to-train-data> \
  --valid_dir <path-to-valid-data> \
  --save_dir <path-to-output-checkpoints> \
  --run_name <experiment-name> \
  --seed <seed>
```

> If you use a CHTC launcher script for this folder, replace the placeholders with the exact paths/arguments from that `.sh` file (same style as done in `Longformer_MLM_pretraining/README.md`).

---

## Model

We use LongformerForSequenceClassification from the transformers package as used by TeClass2 (`Bickmann, Lucas, et al. "Transformer-Based Classification of Transposable Element Consensus Sequences with TEclass2." Biology 15.1 (2025): 59`.)

---

## Config Notes

We used a configuration almost similar to the setup in the original TeClass2 model taken from  `https://github.com/IOB-Muenster/TEclass2/blob/main/config.yml`. We use only one global attention token of BOS.
Similar config files were used for reproducible runs and fair comparisons across models and experiments.

---

## Typical Workflow

1. Configure T settings in `config/`.
2. Prepare train/validation/test inputs in expected data format.
3. Launch training with `train_longformer.py`.
4. Store metrics on validation after the training run has finished.
5. Save the last checkpoint after the chosen number of epochs run for downstream evaluation/inference workflows.

---
