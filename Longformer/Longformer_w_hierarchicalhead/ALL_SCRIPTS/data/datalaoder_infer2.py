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





import re
from dataclasses import dataclass
from pathlib import Path
# from typing import List, Tuple, Union
from typing import Optional, List, Dict, Tuple, Union

import torch
from torch.utils.data import Dataset
from Bio import SeqIO

# =============================================================================
# MAP RULES — supports literal + regex rules
# =============================================================================

_REGEX_META = set(".^$*+?{}[]\\|()")

@dataclass(frozen=True)
class MapRule:
    old: str
    new: str
    is_regex: bool
    pattern: Optional[re.Pattern] = None

def _looks_like_regex(pat: str) -> bool:
    return any(ch in pat for ch in _REGEX_META)

def parse_map_rules(map_rules_str: str) -> List[MapRule]:
    """
    Parse comma-separated rules like:
      "I-Jockey=I,Jockey=I,TcMar-.*=Tc1,^tRNA=tRNA"
    """
    if not map_rules_str or not map_rules_str.strip():
        return []

    rules: List[MapRule] = []
    for pair in map_rules_str.split(","):
        pair = pair.strip()
        if not pair or "=" not in pair:
            continue
        old, new = pair.split("=", 1)
        old, new = old.strip(), new.strip()
        is_regex = _looks_like_regex(old)
        pat = re.compile(old) if is_regex else None
        rules.append(MapRule(old=old, new=new, is_regex=is_regex, pattern=pat))
    return rules

def apply_map_rules_to_field(text: str, rules: List[MapRule]) -> str:
    """
    Apply rules to ONE field only (either order or superfamily).
    This guarantees no cross-field modifications happen.
    """
    for r in rules:
        if r.is_regex:
            text = r.pattern.sub(r.new, text)  # type: ignore[union-attr]
        else:
            text = text.replace(r.old, r.new)
    return text


# =============================================================================
# FASTA PARSER — extract label from header (unchanged)
# =============================================================================

def parse_fasta_label(record_id: str, record_description: str) -> Tuple[str, str]:
    """
    Extract (order, superfamily) from a RepeatMasker-style FASTA header.

    The classification follows '#' and uses '/' as the order/superfamily
    separator. We split ONLY on the FIRST '/', so superfamily names that
    themselves contain '/' (e.g. 'SINE2/tRNA', 'SINE3/5S', 'SINE1/7SL')
    are preserved intact.

    Examples:
        "DNA"                 -> ("DNA", "")
        "DNA/TcMar-Tc1"       -> ("DNA", "TcMar-Tc1")
        "SINE/SINE2/tRNA"     -> ("SINE", "SINE2/tRNA")
        "LTR/ERV/ERV1"        -> ("LTR", "ERV/ERV1")
    """
    if "#" in record_id:
        classification = record_id.split("#", 1)[1]
    elif "#" in record_description:
        parts = record_description.split("#", 1)
        classification = parts[1].split()[0]
    else:
        return ("", "")

    classification = classification.strip()
    if not classification:
        return ("", "")

    # Split ONLY on the first '/'. Anything after the first '/' is the SF,
    # which itself may contain '/' (e.g. 'SINE2/tRNA').
    order, _, sf = classification.partition("/")
    return (order, sf)









# =============================================================================
# DATASET
# =============================================================================

class HierarchicalFASTADataset(Dataset):
    def __init__(
        self,
        fasta_paths,
        label_to_id: dict,
        max_seq_len: int,
        pad_token_id: int = 0,

        # ONE unified rule string, applied independently to order & SF fields
        map_rules_str: str = "",

        min_seq_len: int = 0,
        vocab: dict | None = None,

        # NEW: keep unknown labels instead of skipping them
        keep_unknown: bool = False,
        unknown_node_id: int = 0,  # root is a safe default
    ):
        if vocab is None:
            raise ValueError("vocab must be provided (dict mapping char->token_id)")

        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.vocab = vocab

        self.keep_unknown = keep_unknown
        self.unknown_node_id = unknown_node_id

        # Parse once; apply the same rule list to each field separately.
        self.map_rules = parse_map_rules(map_rules_str)

        if self.map_rules:
            print(f"  Map rules ({len(self.map_rules)}):")
            for r in self.map_rules:
                kind = "REGEX " if r.is_regex else "LITERAL"
                print(f"    [{kind}] '{r.old}' → '{r.new}'")

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

        valid_sequences = []
        valid_node_ids = []
        accessions = []

        # Stored for viz/debug
        mapped_orders = []
        mapped_sfs = []
        had_sfs = []
        sf_was_unknown = []   # NEW: header had SF, but SF label not in tree mapping
        order_was_unknown = [] # NEW: order label not in tree mapping (even after mapping)

        skipped_no_label = 0
        skipped_too_short = 0

        mapped_order_count = 0
        mapped_sf_count = 0

        kept_unknown_count = 0

        for fasta_path in expanded_paths:
            print(f"  Parsing: {fasta_path}")
            for record in SeqIO.parse(str(fasta_path), "fasta"):
                seq_str = str(record.seq).upper()

                if len(seq_str) < min_seq_len:
                    skipped_too_short += 1
                    continue

                # Extract label from header
                order_str, sf_str = parse_fasta_label(record.id, record.description)
                if not order_str:
                    skipped_no_label += 1
                    continue

                had_sf = bool(sf_str)

                # Map each field separately
                mapped_order = apply_map_rules_to_field(order_str, self.map_rules)
                mapped_sf    = apply_map_rules_to_field(sf_str,    self.map_rules) if sf_str else ""

                if mapped_order != order_str:
                    mapped_order_count += 1
                if mapped_sf != sf_str:
                    mapped_sf_count += 1

                # Construct lookup names
                full_name = f"{mapped_order}/{mapped_sf}" if mapped_sf else mapped_order

                # Resolve node_id with fallback logic, but DO NOT DROP unknowns if keep_unknown=True
                resolved_node_id = None

                if full_name in label_to_id:
                    resolved_node_id = label_to_id[full_name]
                    order_unknown = False
                    sf_unknown = False
                elif mapped_order in label_to_id:
                    resolved_node_id = label_to_id[mapped_order]
                    order_unknown = False
                    # SF existed in header but didn't map => unknown superfamily (fallback to order)
                    sf_unknown = had_sf
                else:
                    # Unknown order (and thus unknown full label)
                    order_unknown = True
                    sf_unknown = had_sf  # if SF existed, it's unknown too

                    if self.keep_unknown:
                        resolved_node_id = self.unknown_node_id
                        kept_unknown_count += 1
                    else:
                        # old behavior (drop)
                        continue

                # Keep sample
                valid_sequences.append(seq_str)
                valid_node_ids.append(resolved_node_id)
                accessions.append(record.id)

                mapped_orders.append(mapped_order)
                mapped_sfs.append(mapped_sf)
                had_sfs.append(had_sf)
                order_was_unknown.append(order_unknown)
                sf_was_unknown.append(sf_unknown)

        # Save viz/debug arrays
        self.mapped_orders = mapped_orders
        self.mapped_sfs = mapped_sfs
        self.had_sfs = had_sfs
        self.order_was_unknown = order_was_unknown
        self.sf_was_unknown = sf_was_unknown

        print(f"\n  --- FASTA Parsing Summary ---")
        print(f"  Sequences loaded:        {len(valid_sequences)}")
        print(f"  Order labels remapped:   {mapped_order_count}")
        print(f"  SF labels remapped:      {mapped_sf_count}")
        print(f"  Skipped (no label):      {skipped_no_label}")
        print(f"  Skipped (too short):     {skipped_too_short}")
        print(f"  Kept unknown labels:     {kept_unknown_count} (node_id={self.unknown_node_id})")

        # Pre-encode
        self.all_tokens = self._encode_all(valid_sequences)
        self.all_padding_masks = (self.all_tokens == self.pad_token_id)
        self.all_node_ids = torch.tensor(valid_node_ids, dtype=torch.long)
        self.accessions = accessions

        print(f"  Dataset created: {len(self.all_node_ids)} samples")

    def _encode_all(self, sequence_list):
        encoded = []
        for seq in sequence_list:
            ids = [self.vocab.get(c, self.vocab["X"]) for c in seq[: self.max_seq_len]]
            pad_len = self.max_seq_len - len(ids)
            if pad_len > 0:
                ids = ids + [self.pad_token_id] * pad_len
            encoded.append(ids)
        return torch.tensor(encoded, dtype=torch.long)

    def __len__(self):
        return self.all_tokens.size(0)

    def __getitem__(self, idx):
        return (
            self.all_tokens[idx],
            self.all_padding_masks[idx],
            self.all_node_ids[idx],
        )







class HierarchicalFASTADataset_nounknown(Dataset):
    def __init__(
        self,
        fasta_paths,
        label_to_id: dict,
        max_seq_len: int,
        pad_token_id: int = 0,

        # ONE unified rule string, applied independently to order & SF fields
        map_rules_str: str = "",

        min_seq_len: int = 0,
        vocab: dict | None = None,
    ):
        if vocab is None:
            raise ValueError("vocab must be provided (dict mapping char->token_id)")

        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.vocab = vocab

        # Parse once; apply the same rule list to each field separately.
        self.map_rules = parse_map_rules(map_rules_str)

        if self.map_rules:
            print(f"  Map rules ({len(self.map_rules)}):")
            for r in self.map_rules:
                kind = "REGEX " if r.is_regex else "LITERAL"
                print(f"    [{kind}] '{r.old}' → '{r.new}'")

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

        valid_sequences = []
        valid_node_ids = []
        accessions = []
        mapped_orders = []
        mapped_sfs = []
        had_sfs = []

        skipped_no_label = 0
        skipped_not_in_tree = 0
        skipped_too_short = 0
        mapped_order_count = 0
        mapped_sf_count = 0

        for fasta_path in expanded_paths:
            print(f"  Parsing: {fasta_path}")
            for record in SeqIO.parse(str(fasta_path), "fasta"):
                seq_str = str(record.seq).upper()

                if len(seq_str) < min_seq_len:
                    skipped_too_short += 1
                    continue

                # Extract label from header
                order_str, sf_str = parse_fasta_label(record.id, record.description)
                if not order_str:
                    skipped_no_label += 1
                    continue

                # Apply the SAME rule list to each field independently.
                # Because each field is mapped on its own, a rule keyed to an
                # order name can never accidentally rewrite a superfamily name
                # (and vice-versa), as long as the rule keys are exact/unique
                # to the field they target.
                mapped_order = apply_map_rules_to_field(order_str, self.map_rules)
                mapped_sf    = apply_map_rules_to_field(sf_str,    self.map_rules) if sf_str else ""

                if mapped_order != order_str:
                    mapped_order_count += 1
                if mapped_sf != sf_str:
                    mapped_sf_count += 1

                # Rebuild label for lookup
                lookup_name = f"{mapped_order}/{mapped_sf}" if mapped_sf else mapped_order

                # Look up node ID
                if lookup_name in label_to_id:
                    node_id = label_to_id[lookup_name]
                elif mapped_order in label_to_id:
                    node_id = label_to_id[mapped_order]
                else:
                    skipped_not_in_tree += 1
                    continue

                valid_sequences.append(seq_str)
                valid_node_ids.append(node_id)
                accessions.append(record.id)

                mapped_orders.append(mapped_order)
                mapped_sfs.append(mapped_sf)          # "" if none
                had_sfs.append(bool(sf_str))          # whether fasta header actually had SF
        self.mapped_orders = mapped_orders
        self.mapped_sfs = mapped_sfs
        self.had_sfs = had_sfs

        print(f"\n  --- FASTA Parsing Summary ---")
        print(f"  Sequences loaded:        {len(valid_sequences)}")
        print(f"  Order labels remapped:   {mapped_order_count}")
        print(f"  SF labels remapped:      {mapped_sf_count}")
        print(f"  Skipped (no label):      {skipped_no_label}")
        print(f"  Skipped (not in tree):   {skipped_not_in_tree}")
        print(f"  Skipped (too short):     {skipped_too_short}")

        # Pre-encode
        self.all_tokens = self._encode_all(valid_sequences)
        self.all_padding_masks = (self.all_tokens == self.pad_token_id)
        self.all_node_ids = torch.tensor(valid_node_ids, dtype=torch.long)
        self.accessions = accessions

        print(f"  Dataset created: {len(self.all_node_ids)} samples")

    def _encode_all(self, sequence_list):
        encoded = []
        for seq in sequence_list:
            ids = [self.vocab.get(c, self.vocab["X"]) for c in seq[: self.max_seq_len]]
            pad_len = self.max_seq_len - len(ids)
            if pad_len > 0:
                ids = ids + [self.pad_token_id] * pad_len
            encoded.append(ids)
        return torch.tensor(encoded, dtype=torch.long)

    def __len__(self):
        return self.all_tokens.size(0)

    def __getitem__(self, idx):
        return (
            self.all_tokens[idx],
            self.all_padding_masks[idx],
            self.all_node_ids[idx],
        )






# class HierarchicalFASTADataset(Dataset):
    # def __init__(
    #     self,
    #     fasta_paths,
    #     label_to_id: dict,
    #     max_seq_len: int,
    #     pad_token_id: int = 0,

    #     # NEW: separate rule strings
    #     order_map_rules_str: str = "",
    #     sf_map_rules_str: str = "",

    #     min_seq_len: int = 0,
    #     vocab: dict | None = None,
    # ):
    #     if vocab is None:
    #         raise ValueError("vocab must be provided (dict mapping char->token_id)")

    #     self.max_seq_len = max_seq_len
    #     self.pad_token_id = pad_token_id
    #     self.vocab = vocab

    #     # Parse rules separately (guaranteed to apply only to their field)
    #     self.order_rules = parse_map_rules(order_map_rules_str)
    #     self.sf_rules = parse_map_rules(sf_map_rules_str)

    #     if self.order_rules:
    #         print(f"  ORDER map rules ({len(self.order_rules)}):")
    #         for r in self.order_rules:
    #             kind = "REGEX " if r.is_regex else "LITERAL"
    #             print(f"    [{kind}] '{r.old}' → '{r.new}'")

    #     if self.sf_rules:
    #         print(f"  SUPERFAMILY map rules ({len(self.sf_rules)}):")
    #         for r in self.sf_rules:
    #             kind = "REGEX " if r.is_regex else "LITERAL"
    #             print(f"    [{kind}] '{r.old}' → '{r.new}'")

    #     # Normalize fasta_paths to a list
    #     if isinstance(fasta_paths, (str, Path)):
    #         fasta_paths = [Path(fasta_paths)]
    #     else:
    #         fasta_paths = [Path(p) for p in fasta_paths]

    #     # Expand directories
    #     expanded_paths = []
    #     fasta_extensions = {".fa", ".fasta", ".fna", ".fas"}
    #     for path in fasta_paths:
    #         if path.is_dir():
    #             for ext in fasta_extensions:
    #                 expanded_paths.extend(path.glob(f"*{ext}"))
    #         else:
    #             expanded_paths.append(path)

    #     valid_sequences = []
    #     valid_node_ids = []
    #     accessions = []

    #     skipped_no_label = 0
    #     skipped_not_in_tree = 0
    #     skipped_too_short = 0
    #     mapped_order_count = 0
    #     mapped_sf_count = 0

    #     for fasta_path in expanded_paths:
    #         print(f"  Parsing: {fasta_path}")
    #         for record in SeqIO.parse(str(fasta_path), "fasta"):
    #             seq_str = str(record.seq).upper()

    #             if len(seq_str) < min_seq_len:
    #                 skipped_too_short += 1
    #                 continue

    #             # Extract label from header
    #             order_str, sf_str = parse_fasta_label(record.id, record.description)
    #             if not order_str:
    #                 skipped_no_label += 1
    #                 continue

    #             # Apply mapping strictly per-field
    #             mapped_order = apply_map_rules_to_field(order_str, self.order_rules)
    #             mapped_sf = apply_map_rules_to_field(sf_str, self.sf_rules) if sf_str else ""

    #             if mapped_order != order_str:
    #                 mapped_order_count += 1
    #             if mapped_sf != sf_str:
    #                 mapped_sf_count += 1

    #             # Rebuild label for lookup
    #             lookup_name = f"{mapped_order}/{mapped_sf}" if mapped_sf else mapped_order

    #             # Look up node ID
    #             if lookup_name in label_to_id:
    #                 node_id = label_to_id[lookup_name]
    #             elif mapped_order in label_to_id:
    #                 node_id = label_to_id[mapped_order]
    #             else:
    #                 skipped_not_in_tree += 1
    #                 continue

    #             valid_sequences.append(seq_str)
    #             valid_node_ids.append(node_id)
    #             accessions.append(record.id)

    #     print(f"\n  --- FASTA Parsing Summary ---")
    #     print(f"  Sequences loaded:        {len(valid_sequences)}")
    #     print(f"  Order labels remapped:   {mapped_order_count}")
    #     print(f"  SF labels remapped:      {mapped_sf_count}")
    #     print(f"  Skipped (no label):      {skipped_no_label}")
    #     print(f"  Skipped (not in tree):   {skipped_not_in_tree}")
    #     print(f"  Skipped (too short):     {skipped_too_short}")

    #     # Pre-encode
    #     self.all_tokens = self._encode_all(valid_sequences)
    #     self.all_padding_masks = (self.all_tokens == self.pad_token_id)
    #     self.all_node_ids = torch.tensor(valid_node_ids, dtype=torch.long)
    #     self.accessions = accessions

    #     print(f"  Dataset created: {len(self.all_node_ids)} samples")

    # def _encode_all(self, sequence_list):
    #     encoded = []
    #     for seq in sequence_list:
    #         ids = [self.vocab.get(c, self.vocab["X"]) for c in seq[: self.max_seq_len]]
    #         pad_len = self.max_seq_len - len(ids)
    #         if pad_len > 0:
    #             ids = ids + [self.pad_token_id] * pad_len
    #         encoded.append(ids)
    #     return torch.tensor(encoded, dtype=torch.long)

    # def __len__(self):
    #     return self.all_tokens.size(0)

    # def __getitem__(self, idx):
    #     return (
    #         self.all_tokens[idx],
    #         self.all_padding_masks[idx],
    #         self.all_node_ids[idx],
    #     )








class HierarchicalFASTADatasetInference(Dataset):
    def __init__(
        self,
        fasta_paths,
        label_to_id: dict,
        max_seq_len: int,
        pad_token_id: int = 0,
        map_rules_str: str = "",
        min_seq_len: int = 0,
        vocab: dict | None = None,
        unknown_node_id: int = 0,  # root is safe
    ):
        if vocab is None:
            raise ValueError("vocab must be provided (dict mapping char->token_id)")

        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.vocab = vocab
        self.unknown_node_id = unknown_node_id

        self.map_rules = parse_map_rules(map_rules_str)

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

        seqs = []
        node_ids = []
        accessions = []

        # Keep original mapped labels for reporting
        self.mapped_orders = []
        self.mapped_sfs = []
        self.mapped_full_labels = []
        self.had_sfs = []

        # Diagnostics
        self.label_in_tree = []         # full label in tree
        self.used_order_fallback = []   # full label missing, order used
        self.used_root_fallback = []    # neither full nor order found

        skipped_no_label = 0
        skipped_too_short = 0
        kept_root_fallback = 0
        kept_order_fallback = 0
        kept_full_match = 0

        for fasta_path in expanded_paths:
            print(f"  Parsing: {fasta_path}")
            for record in SeqIO.parse(str(fasta_path), "fasta"):
                seq_str = str(record.seq).upper()

                if len(seq_str) < min_seq_len:
                    skipped_too_short += 1
                    continue

                order_str, sf_str = parse_fasta_label(record.id, record.description)
                if not order_str:
                    skipped_no_label += 1
                    continue

                had_sf = bool(sf_str)

                mapped_order = apply_map_rules_to_field(order_str, self.map_rules)
                mapped_sf = apply_map_rules_to_field(sf_str, self.map_rules) if sf_str else ""

                full_label = f"{mapped_order}/{mapped_sf}" if mapped_sf else mapped_order

                # Choose node_id for loss/metrics, but keep the string label as-is
                if full_label in label_to_id:
                    node_id = label_to_id[full_label]
                    in_tree = True
                    order_fallback = False
                    root_fallback = False
                    kept_full_match += 1
                elif mapped_order in label_to_id:
                    node_id = label_to_id[mapped_order]
                    in_tree = False  # full label not in tree
                    order_fallback = True
                    root_fallback = False
                    kept_order_fallback += 1
                else:
                    node_id = self.unknown_node_id
                    in_tree = False
                    order_fallback = False
                    root_fallback = True
                    kept_root_fallback += 1

                seqs.append(seq_str)
                node_ids.append(node_id)
                accessions.append(record.id)

                self.mapped_orders.append(mapped_order)
                self.mapped_sfs.append(mapped_sf)
                self.mapped_full_labels.append(full_label)
                self.had_sfs.append(had_sf)

                self.label_in_tree.append(in_tree)
                self.used_order_fallback.append(order_fallback)
                self.used_root_fallback.append(root_fallback)

        print(f"\n  --- FASTA Parsing Summary (Inference) ---")
        print(f"  Sequences loaded:          {len(seqs)}")
        print(f"  Full label matched:        {kept_full_match}")
        print(f"  Order fallback used:       {kept_order_fallback}")
        print(f"  Root fallback used:        {kept_root_fallback} (node_id={self.unknown_node_id})")
        print(f"  Skipped (no label):        {skipped_no_label}")
        print(f"  Skipped (too short):       {skipped_too_short}")

        self.all_tokens = self._encode_all(seqs)
        self.all_padding_masks = (self.all_tokens == self.pad_token_id)
        self.all_node_ids = torch.tensor(node_ids, dtype=torch.long)
        self.accessions = accessions

    def _encode_all(self, sequence_list):
        encoded = []
        for seq in sequence_list:
            ids = [self.vocab.get(c, self.vocab["X"]) for c in seq[: self.max_seq_len]]
            pad_len = self.max_seq_len - len(ids)
            if pad_len > 0:
                ids = ids + [self.pad_token_id] * pad_len
            encoded.append(ids)
        return torch.tensor(encoded, dtype=torch.long)

    def __len__(self):
        return self.all_tokens.size(0)

    def __getitem__(self, idx):
        return (
            self.all_tokens[idx],
            self.all_padding_masks[idx],
            self.all_node_ids[idx],
        )