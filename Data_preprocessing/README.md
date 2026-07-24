# Data Preprocessing for TERRIER Labeling (Repbase, MnTEdb, RepetDB)

This folder contains notebooks and a corrected version of the  **TERRIER labeling system** which can be found here - https://github.com/rbturnbull/terrier/blob/main/terrier/data/repbase-to-repeatmasker.toml.

The final outputs of the notebooks can create dataset ready for training (with no k-fold splits) or pretraining into any of our transformer-based architectures.  
The preprocessing flow maps each source dataset to TERRIER labels, filters/standardizes outputs, and exports final files that are then passed to an external pipeline for **stratified k-fold splitting**.

---

## What this preprocessing does

You have 3 source datasets:
- **Repbase**
- **MnTEdb**
- **RepetDB**

For each dataset:
1. A dataset-specific notebook reads and cleans raw data.
2. Labels are mapped using a shared `.toml` TERRIER mapping file.
3. Outputs are normalized into a consistent format.
4. A final filtering notebook produces final mapped/filtered datasets.
5. Final outputs are passed to an external split pipeline for stratified k-fold generation.

---

## Expected folder-level components

This folder is expected to contain:
- Dataset-specific notebooks (`*.ipynb`) for:
  - Repbase TERRIER mapping
  - MnTEdb TERRIER mapping
  - RepetDB TERRIER mapping
- A shared TERRIER mapping config file (`*.toml`)
- A notebook for final filtering/export of mapped datasets

> The dataset-specific notebooks should all reference the same `.toml` label mapping source so label harmonization is consistent across datasets.

---

## Pipeline overview

```text
Raw dataset (Repbase / MnTEdb / RepetDB)
        ↓
Dataset-specific notebook (.ipynb)
        ↓
TERRIER label mapping via shared .toml
        ↓
Mapped dataset output (intermediate)
        ↓
Final filtering notebook (.ipynb)
        ↓
Final filtered mapped datasets
        ↓
External pipeline: stratified k-fold split
```

---

## Label mapping standard (TERRIER)

All three dataset notebooks should:
- load the same `.toml` mapping file,
- apply identical label normalization rules,
- resolve dataset-specific label names into shared TERRIER classes,
- optionally retain provenance columns (original label, mapped label, confidence/notes where relevant).

This ensures a unified label taxonomy before model training.

---

## Recommended output schema

To keep downstream training scripts consistent, final filtered datasets should follow a common schema (example):

- `sequence` (string): primary biological/token sequence
- `mapped_label` (string): TERRIER-mapped class label
- `original_label` (string, optional): source dataset label
- `dataset_name` (string): one of `repbase`, `mntedb`, `repetdb`
- `record_id` (string/int, optional): unique identifier
- additional metadata fields as needed by training/analysis

If pickled dictionary format is used, maintain consistent key/value conventions across all three datasets.

---

## Typical notebook execution order

1. Run **Repbase mapping notebook**
2. Run **MnTEdb mapping notebook**
3. Run **RepetDB mapping notebook**
4. Run **final filtering notebook** on mapped outputs
5. Export final filtered artifacts (e.g., `.pkl` files)
6. Send final artifacts to external k-fold stratified split pipeline

---

## Example handoff artifacts to splitting pipeline

Typical files passed downstream may look like:
- `RepBase*_terrier*.pkl`
- `MnTEdb*_terrier*.pkl`
- `RepetDB*_terrier*.pkl`

(Use your exact naming convention from the notebooks.)

---

## Reproducibility notes

- Use a single source-of-truth `.toml` mapping file version for all runs.
- Version notebook outputs by date/tag (e.g., `*_May13.pkl`) to keep mapping/filter revisions traceable.
- Record:
  - notebook commit/version,
  - `.toml` version,
  - filtering criteria used,
  - dataset counts before/after filtering.

---

## Quality checks before exporting

Before sending to k-fold split pipeline:
- verify no unmapped labels remain (unless explicitly allowed),
- check class distribution after mapping/filtering,
- remove duplicate/empty/invalid sequences,
- confirm output schema is identical across all 3 datasets,
- confirm label names match TERRIER taxonomy exactly.

---

## Downstream usage

The final filtered TERRIER-mapped datasets are consumed by model training pipelines (e.g., Longformer pretraining/finetuning and TeClass2-style training workflows) after external stratified split generation.
