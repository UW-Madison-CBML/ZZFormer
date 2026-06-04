#!/bin/bash
export USER="${USER:-root}"
export LOGNAME="${LOGNAME:-root}"
export HOME="${HOME:-/tmp}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/torch_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg_cache}"
mkdir -p "$HOME" "$TORCHINDUCTOR_CACHE_DIR" "$XDG_CACHE_HOME"

# ─── TorchApp override (your patched apps.py that sets weights_only=False etc.) ───
SUBMIT_DIR="$(pwd)"
git clone https://github.com/rbturnbull/terrier.git "$HOME/terrier-local"
ls -lrt $HOME


cp "prediction_replacement.py" "$HOME/terrier-local/terrier/apps.py"
# ─── Terrier override: evaluate.py (your hardcoded GT-drop behaviour) ───
cp "terrier_evaluate_edit.py" "$HOME/terrier-local/terrier/evaluate.py"

pip install -e "$HOME/terrier-local" --no-deps
python -c "import terrier.apps; print(terrier.apps.__file__)"

cd "$SUBMIT_DIR" || exit 1

mkdir -p "$HOME/pyoverrides"
cp -a /opt/conda/lib/python3.11/site-packages/torchapp "$HOME/pyoverrides/"
cp "$SUBMIT_DIR/torchapp_apps.py" "$HOME/pyoverrides/torchapp/apps.py"
export PYTHONPATH="$HOME/pyoverrides:$PYTHONPATH"


DATASET_NAME="mntedb"







# ─── Train each fold ───
for fold in {0..4}; do
    echo "=== Training Fold $fold ==="

    OUTPUT_DIR="/staging/kkumari/terrsystem/TERRIER/$DATASET_NAME/fold_$fold/"
    mkdir -p "$OUTPUT_DIR"

    SEQBANK="$SUBMIT_DIR/TERRIER_DATA/$DATASET_NAME/fold_${fold}-seqbank_${DATASET_NAME}.sb"
    SEQTREE="$SUBMIT_DIR/TERRIER_DATA/$DATASET_NAME/fold_${fold}-seqtree_${DATASET_NAME}.st"

    terrier-tools train \
        --seqtree "$SEQTREE" \
        --seqbank "$SEQBANK" \
        --max-learning-rate 0.001 \
        --macc 20000000000 \
        --cnn-layers 4 \
        --dropout 0.2479560973202271 \
        --embedding-dim 18 \
        --factor 1.959254226973812 \
        --kernel-size 7 \
        --penultimate-dims 1953 \
        --phi 1.0196823166741456 \
        --max-epochs 100 \
        --validation-partition 1 \
        --save-top-k 1 \
        --test-partition -1 \
        --output-dir "$OUTPUT_DIR"

    echo "=== Fold $fold complete ==="
done






THRESHOLD=0.0
MAP_RULES=""
echo "Using THRESHOLD=$THRESHOLD"
echo "Using MAP_RULES=$MAP_RULES"


DATASET_NAME="mntedb"

for fold in {0..4}; do
  echo "=== Inference/Eval Fold $fold ==="



  INPUT_FASTA="${SUBMIT_DIR}/TERRIER_DATA/_fastas_from_pickles/${DATASET_NAME}/${DATASET_NAME}_fold${fold}_test.fasta"



  FOLD_DIR="/staging/kkumari/terrsystem/TERRIER/${DATASET_NAME}/fold_${fold}"
  CKPT_DIR="${FOLD_DIR}/lightning_logs/version_0/checkpoints"

  CKPT_PATH="${CKPT_DIR}/last.ckpt"


  OUTDIR="/staging/kkumari/terrsystem/TERRIER/validation/${DATASET_NAME}"
  mkdir -p "$OUTDIR"


  PRED_CSV="${OUTDIR}/fold_${fold}.predictions.threshold0.csv"

  terrier \
    --checkpoint "$CKPT_PATH" \
    --input "$INPUT_FASTA" \
    --output-csv "$PRED_CSV" \
    --threshold 0




  METRICS_TXT="${OUTDIR}/fold_${fold}.metrics.threshold${THRESHOLD}.txt"
  terrier-tools evaluate \
      --csv "$PRED_CSV" \
      --threshold "$THRESHOLD" \
      --map "$MAP_RULES" | tee "$METRICS_TXT"


  METRICS_TXT="${OUTDIR}/fold_${fold}.metrics_order_only.threshold${THRESHOLD}.txt"
  terrier-tools evaluate \
      --csv "$PRED_CSV" \
      --threshold "$THRESHOLD" \
      --no-superfamily \
      --map "$MAP_RULES" | tee "$METRICS_TXT"


  CM_EXT=".CSV"

  CM_OUT="${OUTDIR}/fold_${fold}.confusion_matrix.superfamily.threshold${THRESHOLD}${CM_EXT}"

  terrier-tools confusion-matrix \
      --csv "$PRED_CSV" \
      --output "$CM_OUT" \
      --threshold "$THRESHOLD" \
      --map "$MAP_RULES"



  CM_OUT_ORDER="${OUTDIR}/fold_${fold}.confusion_matrix.order_only.threshold${THRESHOLD}${CM_EXT}"

  terrier-tools confusion-matrix \
      --csv "$PRED_CSV" \
      --output "$CM_OUT_ORDER" \
      --threshold "$THRESHOLD" \
      --no-superfamily \
      --map "$MAP_RULES"



done


python3 compute_metrics_validation.py /staging/kkumari/terrsystem/TERRIER $DATASET_NAME  $THRESHOLD