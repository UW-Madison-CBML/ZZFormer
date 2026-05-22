import torch
import torch.nn as nn
import torch.nn.functional as F
from pprint import pprint 

import math

from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import csv

from Bio import SeqIO
from pathlib import Path



DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)



LABEL_MAPPINGS ={'Class': {'ClassI': 0, 'ClassII': 1}, 
    'Subclass': {'LTR': 0, 'Non-LTR': 1, 'Sub1': 2, 'Sub2': 3}, 
    'Order': {'DIRS': 0, 'Helitron': 1, 'LINE': 2, 'Line':2, 'LTR': 3, 'PLE': 4, 'SINE': 5,'Sine': 5, 'TIR': 6}, 
    'Superfamily': {'Bel-Pao': 0, 'CACTA': 1, 'CR1': 2, 'Copia': 3, 'DIRS': 4, 'ERV': 5, 'Gypsy': 6, 'Helitron': 7, 'I': 8, 'ID': 9, 'Jockey': 10, 'L1': 11, 'MULE': 12,'MuLE': 12, 'PIF': 13, 'PLE': 14, 'R2': 15, 'RTE': 16, 'Rex1': 17, 'SINE1/7SL': 18, 'SINE2/tRNA': 19, 'SINE3/5S': 20, 'TcMar': 21, 'hAT': 22,"SINE": 23}}

HIERARCHY_LEVELS = ['Class', 'Subclass', 'Order', 'Superfamily']



vocab = {
        "PAD": 0,
        "A": 1,
        "C": 2,
        "G": 3,
        "T": 4,
        "X": 5
    }
VOCAB_SIZE = len(vocab)
PAD_TOKEN = 0
MASK_TOKEN_ID = VOCAB_SIZE  



ORDER_TO_SUPERFAMILIES={'DIRS': [],
 'Helitron': [],
 'LINE': ['CR1', 'I', 'Jockey', 'L1', 'R2', 'RTE', 'Rex1'],
 'LTR': ['Bel-Pao', 'Copia', 'Gypsy', 'ERV'],
 'PLE': [],
 'SINE': ['ID', 'SINE1/7SL', 'SINE2/tRNA', 'SINE3/5S'],
 'TIR': ['CACTA', 'MULE', 'PIF', 'TcMar', 'hAT']}



# =============================================================================
# MAP RULES — normalize label names before tree lookup
# =============================================================================

def parse_map_rules(map_rules_str: str) -> list[tuple[str, str]]:
    """
    Parse a comma-separated map rules string into a list of (old, new) pairs.

    Example:
        "/I-Jockey=/I,/Jockey=/I,TcMar-Pogo=TcMar,Helitron=RC"
        → [("/I-Jockey", "/I"), ("/Jockey", "/I"), ("TcMar-Pogo", "TcMar")]
    """
    if not map_rules_str or map_rules_str.strip() == "":
        return []

    rules = []
    for pair in map_rules_str.split(","):
        pair = pair.strip()
        if "=" not in pair:
            continue
        old, new = pair.split("=", 1)
        rules.append((old.strip(), new.strip()))
    return rules


def apply_map_rules(label: str, rules: list[tuple[str, str]]) -> str:
    """
    Apply map rules to a label string. Rules are applied in order,
    and each rule does a string replacement on the full label.

    Examples:
        label="LINE/I-Jockey", rule ("/I-Jockey", "/I")  → "LINE/I"
        label="LINE/Jockey",   rule ("/Jockey", "/I")    → "LINE/I"
        label="DNA/TcMar-Pogo", rule ("TcMar-Pogo", "TcMar") → "DNA/TcMar"
        label="Helitron",       rule ("Helitron", "RC")   → "RC"
    """
    for old, new in rules:
        label = label.replace(old, new)
    return label


# =============================================================================
# FASTA PARSER — extract label from header
# =============================================================================

def parse_fasta_label(record_id: str, record_description: str) -> tuple[str, str]:
    """
    Extract (order, superfamily) from a FASTA header.

    Supports RepeatMasker format:
        >accession#Order/Superfamily description
        >accession#Order description          (no superfamily)

    If no '#' is found, tries to parse the description for classification info.

    Returns:
        (order_str, superfamily_str)  where superfamily_str can be ""
    """
    # Try RepeatMasker format: accession#Classification
    if "#" in record_id:
        classification = record_id.split("#", 1)[1]
    elif "#" in record_description:
        # Sometimes the # is in the description, not the id
        parts = record_description.split("#", 1)
        classification = parts[1].split()[0]  # Take first word after #
    else:
        return ("", "")

    # Split classification by "/"
    # e.g., "LINE/L1" → order="LINE", sf="L1"
    # e.g., "LTR/Gypsy" → order="LTR", sf="Gypsy"
    # e.g., "DIRS" → order="DIRS", sf=""
    # e.g., "DNA/TIR/CACTA" → order="TIR", sf="CACTA" (skip Class-level prefix)
    parts = classification.strip().split("/")

    if len(parts) == 1:
        return (parts[0], "")
    elif len(parts) == 2:
        return (parts[0], parts[1])
    elif len(parts) >= 3:
        # e.g., "ClassI/LTR/Gypsy" or "ClassII/DNA/TIR/hAT"
        # Convention: last part is superfamily, second-to-last is order
        return (parts[-2], parts[-1])

    return ("", "")


# =============================================================================
# FASTA DATASET
# =============================================================================

class HierarchicalFASTADataset(Dataset):
    """
    Reads one or more FASTA files, extracts labels from headers,
    applies map rules, and produces (tokens, mask, node_id) tuples.

    FASTA header format expected (RepeatMasker style):
        >accession#Order/Superfamily description
        >accession#Order description

    Args:
        fasta_paths:    list of paths to FASTA files (or a single path)
        label_to_id:    dict mapping "ORDER/SF" or "ORDER" → node_id
        max_seq_len:    maximum sequence length (truncate/pad)
        pad_token_id:   token used for padding
        map_rules_str:  comma-separated remapping rules, e.g.
                        "/I-Jockey=/I,TcMar-Pogo=TcMar,Helitron=RC"
        min_seq_len:    minimum sequence length (skip shorter sequences)
    """

    def __init__(
        self,
        fasta_paths,
        label_to_id: dict,
        max_seq_len: int,
        pad_token_id: int = 0,
        map_rules_str: str = "",
        min_seq_len: int = 0,
    ):
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id

        # Parse map rules
        self.rules = parse_map_rules(map_rules_str)
        if self.rules:
            print(f"  Map rules ({len(self.rules)}):")
            for old, new in self.rules:
                print(f"    '{old}' → '{new}'")

        # Normalize fasta_paths to a list
        if isinstance(fasta_paths, (str, Path)):
            fasta_paths = [Path(fasta_paths)]
        else:
            fasta_paths = [Path(p) for p in fasta_paths]

        # Expand directories
        expanded_paths = []
        fasta_extensions = {".fa", ".fasta", ".fna", ".fas"}
        for path in fasta_paths:
            if path.is_dir():
                for ext in fasta_extensions:
                    expanded_paths.extend(path.glob(f"*{ext}"))
            else:
                expanded_paths.append(path)

        # ---- Parse all FASTA files ----
        valid_sequences = []
        valid_node_ids = []
        accessions = []

        skipped_no_label = 0
        skipped_not_in_tree = 0
        skipped_too_short = 0
        mapped_count = 0

        for fasta_path in expanded_paths:
            print(f"  Parsing: {fasta_path}")
            for record in SeqIO.parse(str(fasta_path), "fasta"):
                seq_str = str(record.seq).upper()

                # Skip short sequences
                if len(seq_str) < min_seq_len:
                    skipped_too_short += 1
                    continue

                # Extract label from header
                order_str, sf_str = parse_fasta_label(record.id, record.description)

                if not order_str:
                    skipped_no_label += 1
                    continue

                # Build full label and apply map rules
                if sf_str:
                    full_label = f"{order_str}/{sf_str}"
                else:
                    full_label = order_str

                mapped_label = apply_map_rules(full_label, self.rules)
                if mapped_label != full_label:
                    mapped_count += 1

                # After mapping, re-split to get order and sf
                if "/" in mapped_label:
                    mapped_parts = mapped_label.split("/", 1)
                    mapped_order = mapped_parts[0]
                    mapped_sf = mapped_parts[1]
                    lookup_name = mapped_label
                else:
                    mapped_order = mapped_label
                    mapped_sf = ""
                    lookup_name = mapped_label

                # Look up node ID
                if lookup_name in label_to_id:
                    node_id = label_to_id[lookup_name]
                elif mapped_order in label_to_id:
                    # Fall back to order-level node
                    node_id = label_to_id[mapped_order]
                else:
                    skipped_not_in_tree += 1
                    continue

                valid_sequences.append(seq_str)
                valid_node_ids.append(node_id)
                accessions.append(record.id)

        # ---- Stats ----
        print(f"\n  --- FASTA Parsing Summary ---")
        print(f"  Sequences loaded:        {len(valid_sequences)}")
        print(f"  Labels remapped:         {mapped_count}")
        print(f"  Skipped (no label):      {skipped_no_label}")
        print(f"  Skipped (not in tree):   {skipped_not_in_tree}")
        print(f"  Skipped (too short):     {skipped_too_short}")

        # ---- Pre-encode everything once ----
        self.all_tokens = self._encode_all(valid_sequences)
        self.all_padding_masks = (self.all_tokens == self.pad_token_id)
        self.all_node_ids = torch.tensor(valid_node_ids, dtype=torch.long)
        self.accessions = accessions  # Keep for per-sample output

        print(f"  Dataset created: {len(self.all_node_ids)} samples")

    def _encode_all(self, sequence_list):
        """Encode every sequence upfront into a single (N, max_seq_len) tensor."""
        encoded = []
        for seq in sequence_list:
            ids = [vocab.get(c, vocab["X"]) for c in seq[: self.max_seq_len]]
            pad_len = self.max_seq_len - len(ids)
            if pad_len > 0:
                ids = ids + [self.pad_token_id] * pad_len
            encoded.append(ids)
        return torch.tensor(encoded, dtype=torch.long)

    def __len__(self):
        return self.all_tokens.size(0)

    def __getitem__(self, idx):
        return (
            self.all_tokens[idx],          # (L,)
            self.all_padding_masks[idx],   # (L,)
            self.all_node_ids[idx],        # scalar
        )






















# =============================================================================
# DATASET — adapted for hierarchical targets
# =============================================================================

class HierarchicalSequenceDataset(Dataset):
    """
    Drop-in replacement for SequenceDataset_nocollate_nomer that produces
    hierarchical node IDs instead of separate order/superfamily labels.

    Pickle format expected: {sequence_string: (order_str, superfamily_str)}
    where superfamily_str can be "" for orders without superfamilies.

    Returns per sample: (tokens, src_key_padding_mask, target_node_id)
    """

    def __init__(
        self,
        sequence_dict,
        label_to_id: dict,
        max_seq_len,
        pad_token_id=0,
        ignore_index=-100,
    ):
        assert ignore_index is not None

        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index

        # ---- Filter and build valid sequence list ----
        valid_sequences = []
        valid_node_ids = []
        skipped = 0

        for seq, (order_str, sf_str) in sequence_dict.items():
            # Build full hierarchical name
            if sf_str and sf_str != "":
                full_name = f"{order_str}/{sf_str}"
            else:
                # Orders without superfamilies (DIRS, Helitron, PLE)
                full_name = order_str

            if full_name in label_to_id:
                node_id = label_to_id[full_name]
            elif order_str in label_to_id:
                node_id = label_to_id[order_str]
            else:
                skipped += 1
                continue

            valid_sequences.append(seq)
            valid_node_ids.append(node_id)

        if skipped > 0:
            print(f"  WARNING: Skipped {skipped} sequences with labels not in classification tree")

        # ---- Pre-encode everything once ----
        self.all_tokens = self._encode_all(valid_sequences)                 # (N, L)
        self.all_padding_masks = (self.all_tokens == self.pad_token_id)     # (N, L)
        self.all_node_ids = torch.tensor(valid_node_ids, dtype=torch.long)  # (N,)

        print(f"  Dataset created: {len(self.all_node_ids)} samples")

    def _encode_all(self, sequence_list):
        """Encode every sequence upfront into a single (N, max_seq_len) tensor."""
        encoded = []
        for seq in sequence_list:
            ids = [vocab.get(c, vocab["X"]) for c in seq[: self.max_seq_len]]
            pad_len = self.max_seq_len - len(ids)
            if pad_len > 0:
                ids = ids + [self.pad_token_id] * pad_len
            encoded.append(ids)
        return torch.tensor(encoded, dtype=torch.long)

    def __len__(self):
        return self.all_tokens.size(0)

    def __getitem__(self, idx):
        return (
            self.all_tokens[idx],          # (L,)
            self.all_padding_masks[idx],   # (L,)
            self.all_node_ids[idx],        # scalar
        )





