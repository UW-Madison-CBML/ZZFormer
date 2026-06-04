#!/bin/bash
export USER="${USER:-root}"
export LOGNAME="${LOGNAME:-root}"
export HOME="${HOME:-/tmp}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/torch_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg_cache}"
mkdir -p "$HOME" "$TORCHINDUCTOR_CACHE_DIR" "$XDG_CACHE_HOME"




rm -rf /tmp/huggingface_cache
mkdir -p /tmp/huggingface_cache
export HF_HOME="/tmp/huggingface_home"


export MPLCONFIGDIR="/tmp/matplotlib_cache"
export WANDB_API_KEY="8b6cf4ca623540253084fe0b0f640583966540d0"
export WANDB_DIR="/tmp/wandb"
export WANDB_CONFIG_DIR="/tmp/wandb"
export WANDB_CACHE_DIR="/tmp/wandb"


mkdir -p /tmp/wandb
export WANDB_DIR="/tmp/wandb"

# Create cache directories
mkdir -p "$TRANSFORMERS_CACHE"
mkdir -p "$HF_HOME"
mkdir -p "$MPLCONFIGDIR"

mkdir -p "$WANDB_DIR" "$WANDB_CONFIG_DIR" "$WANDB_CACHE_DIR"

pip install -U "scikit-learn>=0.22"
python -c "import sklearn; print(sklearn.__version__)"

SUBMIT_DIR="$(pwd)"  # This is where the job was submitted from

WORKDIR="$SUBMIT_DIR/TERL_Code"

cd "$WORKDIR" || exit 1


BASE_DIR="$WORKDIR"



# DATASET_NAMES=("mntedb" "repetdb" "repbase")

DATASET_NAME="repbase"

INPUT_FASTASS=("${SUBMIT_DIR}/Test_files/Drosophila_melanogaster.fasta" "${SUBMIT_DIR}/Test_files/Oryza_sativa.fasta" "${SUBMIT_DIR}/Test_files/Mus_musculus.fasta"  "${SUBMIT_DIR}/Test_files/Homo_sapiens.fasta")



for INPUT_FASTA in "${INPUT_FASTASS[@]}"; do
    OUTNAME="$(basename "$INPUT_FASTA" .fasta)"
    OUT_DIR_BASE="/staging/kkumari/TERL/inference/GT/"
    mkdir -p "${OUT_DIR_BASE}"


    python build_groundtruth_pickle.py \
        -i $INPUT_FASTA \
        -o "${OUT_DIR_BASE}/${OUTNAME}.pkl" \
        --alias "I-Jockey=I,Jockey=I,TcMar-Pogo=TcMar,TcMar-Tc1=TcMar,CMC-Transib=CMC,R1-LOA=R1,hAT-hobo=hAT,hAT-Tip100=hAT,CMC-EnSpm=CMC,RC=Helitron,Mariner-Tc1=TcMar,Dirs=DIRS,Pao=Bel-Pao,DNA=TIR,CMC-.*=CACTA,Tc1-.*=Tc1,hAT-.*=hAT,ERV.*=ERV,L1-.*=L1,PIF-Harbinger=PIF,Crypton-.*=Crypton,RTE-.*=RTE,Retroposon=LINE,I-Jockey=I,MULE-.*=MULE,R1-.*=R1,Penelope=PLE,Unknown="



    for fold in {0..4}; do



        OUT_DIR=/staging/kkumari/TERL/inference/$DATASET_NAME/SF/${fold}
        OUT_PREFIX="${OUT_DIR}/fold${fold}_sf_${OUTNAME}.fasta"


        python eval_predictions.py \
            -p "$OUT_PREFIX" \
            -g "${OUT_DIR_BASE}/${OUTNAME}.pkl" \
            --level superfamily \
            --pred-alias 'PIF-Harbinger=PIF,Dirs=DIRS,Mariner=TcMar,Jockey=I' \
            --model-classes "SINE1,R2,TcMar,SINE2,RTE,SINE,SINE3,Rex1,hAT,Bel-Pao,Gypsy,PIF,I,Copia,Jockey,MULE,CR1,CACTA,L1,ERV" \
            --gt-alias "tRNA-.*=SINE2,tRNA.*=SINE2,5S.*=SINE3,5S-.*=SINE3,Alu=SINE1,B2=SINE2,B4=SINE2,MIR=SINE2,ID=SINE2,L1-.*=L1" \
            -o "${OUT_DIR}/${OUTNAME}_metrics_superfamily.txt"





        OUT_DIRRR=/staging/kkumari/TERL/inference/${DATASET_NAME}/Order/${fold}
        OUT_PREFIX="${OUT_DIRRR}/fold${fold}_order_${OUTNAME}.fasta"

        python eval_predictions.py \
            -p "$OUT_PREFIX" \
            -g "${OUT_DIR_BASE}/${OUTNAME}.pkl" \
            --level order \
            --pred-alias 'Dirs=DIRS,RC=Helitron,Jockey=I' \
            --model-classes "SINE,LINE,DIRS,Helitron,TIR,LTR,PLE" \
            --gt-alias 'SINE2=SINE,SINE3=SINE,SINE1=SINE,Dirs=DIRS,RC=Helitron' \
            -o "${OUT_DIRRR}/${OUTNAME}_metrics_order.txt"

    
    done
done






#implementation in TERRIER
# "/I-Jockey=/I,/Jockey=/I,TcMar-Pogo=TcMar,TcMar-Tc1=TcMar,Mariner/Tc1:TcMar,CMC-Transib=CMC,R1-LOA=R1,hAT-hobo=hAT,hAT-Tip100=hAT,CMC-EnSpm=CMC,RC=Helitron,Mariner-Tc1=TcMar,Dirs=DIRS,Pao=Bel-Pao,DNA=TIR,CMC-.*=CACTA,Tc1-.*=Tc1,hAT-.*=hAT,ERV.*=ERV,PIF-Harbinger=Harbinger,Crypton-.*=Crypton,RTE-.*=RTE,Retroposon=LINE,^tRNA=SINE/tRNA,SINE/tRNA-.*=SINE/tRNA,SINE/5S-.*=SINE/5S,SINE/Alu=SINE/7SL,SINE/B2=SINE/tRNA,SINE/B4=SINE/tRNA,SINE/MIR=SINE/tRNA,SINE/ID=SINE/tRNA,I-Jockey=I,Jockey.*=/I,MULE-.*=MULE,R1-.*=R1"


# implementation in my own classifier, but for us Jockey is Jockey, not "I", and SINE2/tRNA is a superfamily! -
# Corrected: "I-Jockey=I,TcMar-Pogo=TcMar,TcMar-Tc1=TcMar,CMC-Transib=CMC,R1-LOA=R1,hAT-hobo=hAT,hAT-Tip100=hAT,CMC-EnSpm=CMC,RC=Helitron,Mariner-Tc1=TcMar,Dirs=DIRS,Pao=Bel-Pao,DNA=TIR,CMC-.*=CACTA,Tc1-.*=Tc1,hAT-.*=hAT,ERV.*=ERV,PIF-Harbinger=PIF,Crypton-.*=Crypton,RTE-.*=RTE,Retroposon=LINE,I-Jockey=I,MULE-.*=MULE,R1-.*=R1,tRNA-.*=SINE2/tRNA,5S-.*=SINE3/5S,Alu=SINE1/7SL,B2=SINE2/tRNA,B4=SINE2/tRNA,MIR=SINE2/tRNA,ID=SINE2/tRNA"
#Wrong- ^tRNA=SINE2/tRNA,SINE/tRNA-.*=SINE2/tRNA,SINE/5S-.*=SINE3/5S,SINE/Alu=SINE1/7SL,SINE/B2=SINE2/tRNA,SINE/B4=SINE2/tRNA,SINE/MIR=SINE2/tRNA,SINE/ID=SINE2/tRNA"


#Implemented in TERL and DeepTE-
# "I-Jockey=I,TcMar-Pogo=TcMar,TcMar-Tc1=TcMar,CMC-Transib=CMC,R1-LOA=R1,hAT-hobo=hAT,hAT-Tip100=hAT,CMC-EnSpm=CMC,RC=Helitron,Mariner-Tc1=TcMar,Dirs=DIRS,Pao=Bel-Pao,DNA=TIR,CMC-.*=CACTA,Tc1-.*=Tc1,hAT-.*=hAT,ERV.*=ERV,PIF-Harbinger=PIF,Crypton-.*=Crypton,RTE-.*=RTE,Retroposon=LINE,I-Jockey=I,MULE-.*=MULE,R1-.*=R1,^tRNA=SINE2,SINE/tRNA-.*=SINE2,SINE/5S-.*=SINE3,SINE/Alu=SINE1,SINE/B2=SINE2,SINE/B4=SINE2,SINE/MIR=SINE2,SINE/ID=SINE2"
#^tRNA=SINE2,SINE/tRNA-.*=SINE2,SINE/5S-.*=SINE3,SINE/Alu=SINE1,SINE/B2=SINE2,SINE/B4=SINE2,SINE/MIR=SINE2,SINE/ID=SINE2"



DATASET_NAME="repbase"

# superfamily-level summary across folds
python aggregate_folds.py --root "/staging/kkumari/TERL/inference/$DATASET_NAME/SF/" --level superfamily --out "/staging/kkumari/TERL/inference/$DATASET_NAME/SF_summary.tsv"

# order-level summary
python aggregate_folds.py --root "/staging/kkumari/TERL/inference/$DATASET_NAME/Order/" --level order --out "/staging/kkumari/TERL/inference/$DATASET_NAME/Order_summary.tsv"




















# =========================================================
# 2) INFERENCE on your FASTA files using the saved model

#7SL=SINE1/7SL,tRNA=SINE2/tRNA,5S=SINE3/5S,TcMar-Mariner=TcMar
# True label distribution : [('ERVK', 353), ('L1', 320), ('ERV1', 145), ('ERVL', 119), ('ERVL-MaLR', 83), ('hAT-Charlie', 82), ('hAT', 65), ('TcMar-Tigger', 55), ('Alu', 33), ('CR1', 26), ('B2', 25), ('Gypsy', 25), ('L2', 15), ('B4', 8), ('TcMar', 8), ('TcMar-Tc2', 8), ('Crypton', 7), ('MIR', 6), ('hAT-Blackjack', 6), ('PIF', 6), ('Helitron', 5), ('hAT-Ac', 5), ('RTE-X', 4), ('TcMar-Mariner', 4), ('ID', 4), ('L1-dep', 3), ('DIRS', 3), ('Crypton-A', 2), ('PiggyBac', 2), ('Merlin', 2), ('Kolobok', 2), ('RTE-BovB', 2), ('MULE-MuDR', 2), ('hAT-Tag1', 2), ('7SL', 1), ('5S-Deu-L2', 1), ('tRNA-Deu', 1), ('tRNA', 1), ('Dong-R4', 1), ('tRNA-RTE', 1), ('hAT-hAT19', 1), ('L1-Tx1', 1), ('Penelope', 1), ('I', 1), ('tRNA-Deu-L2', 1), ('centromeric', 1), ('Y-chromosome', 1)]
# Pred label distribution : [('ERV', 534), ('L1', 233), ('Gypsy', 222), ('hAT', 164), ('TcMar', 73), ('.fa', 57), ('SINE2', 57), ('PIF', 49), ('Copia', 32), ('MULE', 11), ('CR1', 6), ('Bel-Pao', 6), ('RTE', 5), ('CACTA', 1)]

# True label distribution : [('ERV1', 322), ('L1', 198), ('ERVL', 128), ('Alu', 121), ('hAT-Charlie', 85), ('TcMar-Tigger', 85), ('hAT', 66), ('ERVL-MaLR', 59), ('ERVK', 47), ('CR1', 26), ('Gypsy', 25), ('L2', 20), ('centromeric', 15), ('MIR', 10), ('TcMar', 8), ('TcMar-Tc2', 8), ('Helitron', 7), ('Crypton', 7), ('MULE-MuDR', 7), ('hAT-Blackjack', 6), ('TcMar-Mariner', 6), ('PiggyBac', 6), ('PIF', 6), ('Y-chromosome', 5), ('hAT-Ac', 5), ('SVA', 5), ('RTE-X', 4), ('acromeric', 3), ('Merlin', 3), ('DIRS', 3), ('Crypton-A', 2), ('Kolobok', 2), ('subtelomeric', 2), ('RTE-BovB', 2), ('hAT-Tag1', 2), ('Copia', 1), ('telomeric', 1), ('5S-Deu-L2', 1), ('tRNA-Deu', 1), ('tRNA', 1), ('Dong-R4', 1), ('tRNA-RTE', 1), ('hAT-hAT19', 1), ('L1-Tx1', 1), ('Penelope', 1), ('I', 1), ('tRNA-Deu-L2', 1)]
# Pred label distribution : [('ERV', 496), ('hAT', 186), ('Gypsy', 174), ('L1', 150), ('TcMar', 98), ('.fa', 81), ('SINE2', 47), ('PIF', 36), ('Copia', 23), ('Bel-Pao', 9), ('MULE', 6), ('CR1', 5), ('RTE', 4), ('CACTA', 3)]


# True label distribution : [('Gypsy', 35), ('Copia', 26), ('Helitron', 5), ('L1', 4), ('CMC', 3), ('hAT', 2)]
# Pred label distribution : [('Gypsy', 24), ('Copia', 14), ('.fa', 9), ('CACTA', 7), ('TcMar', 7), ('MULE', 5), ('L1', 3), ('CR1', 2), ('hAT', 2), ('PIF', 2)]

# True label distribution : [('Gypsy', 352), ('I', 101), ('Pao', 68), ('R1', 45), ('TcMar', 22), ('Copia', 18), ('P', 17), ('Helitron', 15), ('CR1', 11), ('CMC', 10), ('hAT', 1), ('R2', 1)]
# Pred label distribution : [('Gypsy', 249), ('TcMar', 113), ('.fa', 54), ('hAT', 53), ('PIF', 35), ('MULE', 35), ('L1', 30), ('Bel-Pao', 30), ('Copia', 25), ('SINE2', 19), ('Jockey', 7), ('ERV', 5), ('CR1', 3), ('RTE', 3)]

# =========================================================
