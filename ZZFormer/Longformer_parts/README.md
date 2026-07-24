# Longformer_parts

This folder contains training pipelines for Longformer-based classification with multiple training variants:
- without pretraining weights - this is TeClass2; please refer to the folder `TeClass2`
- with pretraining weights (hierarchical setup),
- with pretraining weights (non-hierarchical setup).

The code is organized as follows:
- `model/` for model architecture definitions,
- `data/` for dataloader/dataset logic,
- `train_*.py` scripts for experiment entry points,
- `utils.py` for metric computation and shared utilities.

---

## Folder Structure

```text
Longformer_parts/
├── config/
├── data/
├── model/
├── train_longformer_withpretrnwts.py
├── train_longformer_withpretrnwts_nothierarchical.py
├── summarize_folds.py
├── utils.py
└── visualize_umap.py
```
**Input Data:**  
- For all of these models, the same input pickle files were used. These pickle files contain a dictionary of the format - {seq: label} where label is of the format - ''order/superfamily''.
---

## Training Scripts

### `train_longformer_withpretrnwts.py`
**Model 1:** Train Longformer along with LongformerForMaskedLM pretrained weights with a hierarchical head.

**Model**  
- Longformer model with weight loading path passed via args/config, plus task-specific head.
- Long-sequence encoder was initialized from MLM-pretrained Longformer weights, followed by hierarchical softmax after the classification layers.

**Run (template):**
```bash
python train_longformer_withpretrnwts.py \
  --config <path-to-config> \
  --pretrained_model_path <checkpoint> \
  --train_dir <path to train_pickle> \
  --val_dir <path to val_pickle> \
  --save_dir <path to save directory> \
  --run_name <name of the run> \
  --seed <random seed> \
  --wandb_project <optional wandb project name> \
  --wandb_team <optional wandb team name> \
  --wandb_dir <optional wandb directory>
  
```


---

### `train_longformer_withpretrnwts_nothierarchical.py`
**Model 2:** Train Longformer with the pretrained weights with a non-hierarchical classification head

**Model**  
- Longformer + with a non-hierarchical classification head, initialized with the pretraining weights. 
- Same base long-sequence transformer, but with non-hierarchical representation/forward pass choices.

**Input Data**  
- Non-hierarchical dataloader from `data/` (single-pass sequence packing strategy).

**Run (template):**
```bash
python train_longformer_withpretrnwts_nothierarchical.py \
  --config <path-to-config> \
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


---

## Model Package (`model/`)

The `model/` folder contains the hierarchical classifier head on top of the LongformerModel architecture  used by the `train_longformer_withpretrnwts.py` scripts.  

> Check imports at the top of each training script to identify exactly which model class is used in each variant.

---

## Data Package (`data/`)

The dataloader is contained within each train_<>.py as the class 'NucleotideClassificationDataset'; this does the following:
- Truncating or padding sequences to a length of 1024, and then adding the BOS (token number 6) and EOS (token number 7) 
- tokenization/encoding using the vocab definition, which is consistent across different models used - {"PAD":0, "a":1, "c":2, "g":3, "t":4, "x":5, "BOS": 6, "EOS": 7}
- collate/batching according to the batch size requested.

---

## Metrics & Utilities (`utils.py`)

`utils.py` contains shared helpers and metric computation used during training/evaluation, such as:
- Macro classification metrics (for precision/recall/F1/accuracy as implemented),


---

## Typical Workflow

1. Prepare splits of the dataset and config.
2. Pick training variant:
   - no pretrain: `train_longformer_WOpretrain.py`
   - pretrain hierarchical: `train_longformer_withpretrnwts.py`
   - pretrain non-hierarchical: `train_longformer_withpretrnwts_nothierarchical.py`
3. Run training for each fold; this can be done using the Docker image - kritikakumari22/tda_seqemb:zzformer_transformeronly5 and installing packages hierarchicalsoftmax, scikit-learn on top of it using pip. To run it on an HTCondor cluster, refer to the CHTC_scripts_<> folders to understand how it can be run in a bash environment
4. Aggregate fold results via `summarize_folds.py`.
5. Use `visualize_umap.py` for embedding visualization/analysis using the following arguments-
```bash
python visualize_umap.py \
  --config "$WORKDIR"/config/longformer_config.yml \
  --train_file $TRAIN_FILE \
  --test_file   $TEST_FILE \
  --model_dir "$OUTPUT_DIR"/longformer_fold${fold}_${DATASET_NAME}_longformerHierarprtrn_${fold}.pt \
  --save_dir "$OUTPUT_DIR2" \
  --run_name $run_name \
  --DPI 1000
```
---

## Notes

- Keep config paths and pretrained checkpoint paths explicit in command line or config files.
- Ensure tokenizer/model max sequence settings are aligned with MLM pretraining and this finetuning training.
- Use the same fold definitions across variants for fair comparison.






---
