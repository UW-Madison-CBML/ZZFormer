from pathlib import Path
from Bio import SeqIO
from seqbank import SeqBank
from terrier.repeatmasker import get_verbatim_classification
from hierarchicalsoftmax import SoftmaxNode
from corgi.seqtree import SeqTree
import toml
import sys
import os
# Load classification mapping
# mapping_path = Path("terrier/data/repbase-to-repeatmasker.toml")  # adjust to your install path
mapping_path = Path("repbase-to-repeatmasker.toml")  # adjust to your install path
with open(mapping_path) as f:
    mapping = toml.load(f)
datasetname=sys.argv[1] 
current_dir = os.getcwd()
output_dir = f"{current_dir}/train/"
os.makedirs(output_dir, exist_ok=True)


for fold in range(0, 5):
    train_fasta = Path(f"conv_fold_{fold}_train_{datasetname}.fasta")
    test_fasta = Path(f"conv_fold_{fold}_test_{datasetname}.fasta")

    # Combine train and test FASTA files into a single file
    
    combined_fasta = Path(f"{output_dir}/fold_{fold}_{datasetname}_combined.fasta")
    with open(combined_fasta, 'w') as outfile:
        SeqIO.write(SeqIO.parse(train_fasta, 'fasta'), outfile, 'fasta')
        SeqIO.write(SeqIO.parse(test_fasta, 'fasta'), outfile, 'fasta')

    # Build SeqBank with all sequences
    sb = SeqBank(path=f"{output_dir}/fold_{fold}-seqbank_{datasetname}.sb", write=True)
    sb.add_files([train_fasta, test_fasta], format="fasta")

    # Build SeqTree with controlled partition assignment
    classification_tree = SoftmaxNode(name="root")
    classification_nodes = {}
    seqtree = SeqTree(classification_tree)
    # Add counters for debugging
    total_processed = 0
    added_count = 0
    skipped_count = 0
    classification_counts = {} 

    for fasta_path, partition in [(train_fasta, 0), (test_fasta, 1)]:
        with open(fasta_path) as f:
            for record in SeqIO.parse(f, "fasta"):
                accession = record.id
                classification = get_verbatim_classification(fasta_path, record)
                # print(f"Processing {accession}: original classification = {classification}")

                if classification in mapping:
                    classification = mapping[classification]
                    # print(f"Mapped to: {classification}")
                if classification not in mapping.values() or classification == "Unknown":
                    print("Warning: Unmapped or unknown classification for accession {}: {}".format(accession, classification))
                    continue

                # Build tree nodes
                if classification not in classification_nodes:
                    components = classification.split("/")
                    repeat_type = components[0]
                    repeat_subtype = components[1] if len(components) > 1 else ""
                    if repeat_type not in classification_nodes:
                        classification_nodes[repeat_type] = SoftmaxNode(
                            repeat_type, parent=classification_tree
                        )
                    if repeat_subtype:
                        classification_nodes[classification] = SoftmaxNode(
                            repeat_subtype, parent=classification_nodes[repeat_type]
                        )

                node = classification_nodes[classification]
                try:
                    # partition 0 = train, partition 1 = validation
                    seqtree.add(accession, node, partition)
                    added_count += 1
                    if classification not in classification_counts:
                        classification_counts[classification] = 0
                    classification_counts[classification] += 1
                    # print(f"Successfully added {accession} to {classification}")
                
                except Exception as err:
                    print(f"Failed to add {accession} to {classification}: {err}")
                    skipped_count += 1


    # Print summary
    print(f"\nSummary for fold {fold}:")
    print(f"Total processed: {total_processed}")
    print(f"Successfully added: {added_count}")
    print(f"Skipped: {skipped_count}")
    print("Counts per classification:")
    for cls, count in sorted(classification_counts.items()):
        print(f"  {cls}: {count}")

    seqtree.save(Path(f"{output_dir}/fold_{fold}-seqtree_{datasetname}.st"))
    print(f"Fold {fold}: saved seqtree with {len(seqtree)} sequences")









from pathlib import Path
import numpy as np
import torch
from terrier.apps import Terrier

# Set these to your real paths
ckpt = Path("/staging/kkumari/TERRIER/repbase/fold_0/lightning_logs/version_0/checkpoints/last.ckpt")
seqtree = Path("./repbase/train/fold_0-seqtree_repbase.st")  # change
fasta = "./Test_files/Drosophila_melanogaster.fasta"

app = Terrier()

# Force Terrier to load THIS tree (even if deprecated, we pass it through setup)
# Note: this may still not be used; we will print what it ended up with.
app.setup(seqtree=seqtree, input=fasta, checkpoint=ckpt)

# Load checkpoint (your patched torchapp should do weights_only=False; but we avoid TorchApp loader here)
from torchapp.apps import TorchApp
module_class = app.module_class()
module = module_class.load_from_checkpoint(str(ckpt), weights_only=False, map_location="cpu")
module.eval()

# Build dataloader the same way predict does (Terrier likely has prediction_dataloader)
dl = app.prediction_dataloader(module, input=fasta, batch_size=1, max_length=5000, min_length=128)

# Run one prediction batch to get the raw tensor shape
batch = next(iter(dl))
# batch may be tuple/list
inputs = batch[: module.input_count] if isinstance(batch, (tuple, list)) else batch
with torch.inference_mode():
    out = module.model(*inputs)  # may be tensor or tuple

print("=== MODEL OUTPUT SHAPES ===")
if isinstance(out, torch.Tensor):
    print("out:", tuple(out.shape))
else:
    print("out type:", type(out))
    try:
        for i, t in enumerate(out):
            print(f"out[{i}]:", tuple(t.shape))
    except Exception as e:
        print("could not iterate out:", e)

# Now print tree-derived category count
tree = getattr(app, "classification_tree", None)
print("\n=== TREE ===")
print("classification_tree present:", tree is not None)
if tree is not None:
    non_root = [n for n in tree.node_list if not n.is_root]
    print("non-root nodes:", len(non_root))
    # print the last few to see what's extra
    print("last 5 nodes:", [str(n) for n in non_root[-5:]])




















python3 - <<'PY'
USER=condor LOGNAME=condor HOME=/tmp XDG_CACHE_HOME=/tmp/xdg_cache \
TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_cache TORCH_COMPILE_CACHE_DIR=/tmp/torchcompile_cache \
TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1 TORCHINDUCTOR_DISABLE=1 \
python3 - <<'PY'
from pathlib import Path
from terrier.apps import Terrier

ckpt_path = Path("/staging/kkumari/TERRIER/repbase/fold_0/lightning_logs/version_0/checkpoints/last.ckpt")  # change
seqtree = Path("./repbase/train/fold_0-seqtree_repbase.st")  # change
fasta = Path("/var/lib/condor/execute/slot3/dir_8751/scratch/Test_files/Drosophila_melanogaster.fasta")  # change

app = Terrier()

# IMPORTANT: load the LightningModule (trusted checkpoint)
module_class = app.module_class()
module = module_class.load_from_checkpoint(str(ckpt_path), weights_only=False, map_location="cpu")

# This call is what likely builds app.classification_tree
dl = app.prediction_dataloader(
    module,
    input=str(fasta),
    seqtree=str(seqtree),
    batch_size=1,
    max_length=5000,
    min_length=128,
)

tree = getattr(app, "classification_tree", None)
print("classification_tree set:", tree is not None)
if tree is not None:
    non_root = [n for n in tree.node_list if not n.is_root]
    print("non-root nodes:", len(non_root))
    # print all names
    def node_lineage_string(node) -> str:
        return "/".join([str(n) for n in node.ancestors[1:]] + [str(node)])
    for i, node in enumerate(non_root):
        print(i, node_lineage_string(node))
PY

PY