import torch
from torch.utils.data import Dataset
import numpy as np

VOCAB = {
    "PAD":  0, "a": 1, "c": 2, "g": 3, "t": 4,
    "x":    5, "BOS": 6, "EOS": 7, "MASK": 8,
}
PAD_TOKEN_ID  = VOCAB["PAD"]
BOS_TOKEN_ID  = VOCAB["BOS"]
EOS_TOKEN_ID  = VOCAB["EOS"]
UNK_TOKEN_ID  = VOCAB["x"]



# Separate kmer
class TopoDataset(Dataset):
    """
    Dataset for HierarchicalLongformerClassifier using unpacked data arrays.
    Handles both flat lists and chunked list-of-lists inputs.
    Stores each k-mer image tensor in its own individual key (e.g., '4mer_image').
    """
    def __init__(
        self,
        data_dict,
        max_seq_len,
        k_mers=(4, 8, 14, 20),
        mask=False,
    ):
        self.max_seq_len = max_seq_len
        self.mask        = mask
        self.k_mers      = list(k_mers)
        # 1. Unify and flatten metadata fields
        self.sequences = data_dict.get('sequences', [])
        self.labels    = data_dict.get('labels', [])
        self.label_ids = data_dict.get('label_ids', [])
        # 2. Process and stack k-mer images along axis 0
        self.pi_arrays = {}
        for k in self.k_mers:
            key = f'{k}mer'
            if key in data_dict and len(data_dict[key]) > 0:
                raw_data = data_dict[key]
                # Case A: Already a list of numpy matrices from .extend() -> stack/convert to array
                if isinstance(raw_data, list):
                    if isinstance(raw_data[0], np.ndarray):
                        if raw_data[0].ndim == 3:
                            self.pi_arrays[k] = np.stack(raw_data, axis=0)
                        elif raw_data[0].ndim == 4:
                            self.pi_arrays[k] = np.concatenate(raw_data, axis=0)
                    else:
                        self.pi_arrays[k] = np.array(raw_data)
                # Case B: Already a unified Numpy array
                elif isinstance(raw_data, np.ndarray):
                    self.pi_arrays[k] = raw_data
        # Sanity Check Validations
        assert len(self.pi_arrays) == len(self.k_mers), (
            f"Missing k-mer keys. Expected {self.k_mers}, found keys: {list(self.pi_arrays.keys())}. "
            "Ensure train_data['4mer'].extend(...) isn't commented out in your fold loop!"
        )
        assert len(self.sequences) == len(self.label_ids) == len(self.labels), (
            f"Metadata length mismatch: sequences ({len(self.sequences)}), "
            f"label_ids ({len(self.label_ids)}), labels ({len(self.labels)})"
        )
        # Tokenization & Target Tensors
        self.input_ids, self.attention_masks = self._tokenize_all()
        self.targets = torch.tensor(self.label_ids, dtype=torch.long)
        print(f"  TopoDataset: {len(self.sequences)} samples | L={max_seq_len} | k-mers={self.k_mers}")
    def _flatten_list(self, input_list):
        """Flattens a list if it contains sublists (chunks), otherwise returns it as-is."""
        if not input_list:
            return []
        if isinstance(input_list[0], list):
            return [item for sublist in input_list for item in sublist]
        return input_list
    def _tokenize_all(self):
        N, L = len(self.sequences), self.max_seq_len
        input_ids      = torch.full((N, L), PAD_TOKEN_ID, dtype=torch.long)
        attention_mask = torch.zeros((N, L), dtype=torch.long)
        body_max = L - 2
        for i, seq in enumerate(self.sequences):
            seq  = str(seq).lower()
            body = [VOCAB.get(c, UNK_TOKEN_ID) for c in seq[:body_max]]
            ids  = [BOS_TOKEN_ID] + body + [EOS_TOKEN_ID]
            input_ids[i, : len(ids)]      = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, : len(ids)] = 1
        return input_ids, attention_mask
    def _get_pi_tensor(self, k_mer, idx):
        # Convert HWC numpy array to PyTorch CHW tensor
        pi = self.pi_arrays[k_mer][idx]  # shape: (H, W, C)
        return torch.as_tensor(pi, dtype=torch.float32)#.permute(2, 0, 1).contiguous()
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        item = {
            "input_ids":       self.input_ids[idx],
            "attention_mask":  self.attention_masks[idx],
            "target_node_ids": self.targets[idx],
            "labels":          self.labels[idx],
            "sequence":        seq,
        }
        # Save each k-mer image in its own key: e.g. '4mer_image', '8mer_image'
        images = []
        for k in self.k_mers:
            img_tensor = self._get_pi_tensor(k, idx)
            item[f"{k}mer_image"] = img_tensor
            if self.mask:
                images.append(img_tensor)
        if self.mask:
            item["topo_mask"] = torch.tensor(
                [1.0 if torch.any(img != 0) else 0.0 for img in images], 
                dtype=torch.float32
            )
        return item
