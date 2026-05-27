import torch
import torch.nn as nn
import torch.nn.functional as F
from pprint import pprint 

import math

from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import csv
import pickle




DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)

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


LABEL_MAPPINGS ={'Class': {'ClassI': 0, 'ClassII': 1}, 
    'Subclass': {'LTR': 0, 'Non-LTR': 1, 'Sub1': 2, 'Sub2': 3}, 
    'Order': {'DIRS': 0, 'Helitron': 1, 'LINE': 2, 'Line':2, 'LTR': 3, 'PLE': 4, 'SINE': 5,'Sine': 5, 'TIR': 6}, 
    'Superfamily': {'Bel-Pao': 0, 'CACTA': 1, 'CR1': 2, 'Copia': 3, 'DIRS': 4, 'ERV': 5, 'Gypsy': 6, 'Helitron': 7, 'I': 8, 'ID': 9, 'Jockey': 10, 'L1': 11, 'MULE': 12,'MuLE': 12, 'PIF': 13, 'PLE': 14, 'R2': 15, 'RTE': 16, 'Rex1': 17, 'SINE1/7SL': 18, 'SINE2/tRNA': 19, 'SINE3/5S': 20, 'TcMar': 21, 'hAT': 22,"SINE": 23}}

HIERARCHY_LEVELS = ['Class', 'Subclass', 'Order', 'Superfamily']

def read_tsv(file_path):
    data = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            data.append(row)
    return data







class SequenceDataset_with_topology(Dataset):
    def __init__(
        self,
        sequence_dict,
        max_seq_len,
        pad_token_id=0,
        ignore_index=-100,
        mer2_path=None,
        mer4_path=None,
        mer8_path=None,
    ):
        assert ignore_index is not None
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index

        sequences = list(sequence_dict.keys())

        # ---- Tokens & Labels ----
        self.all_tokens = self._encode_all(sequences)
        self.all_padding_masks = (self.all_tokens == self.pad_token_id)
        self.all_order_labels = self._encode_labels(sequences, sequence_dict, 0, 'Order')
        self.all_superfamily_labels = self._encode_labels(sequences, sequence_dict, 1, 'Superfamily')

        # ---- K-mer latents: separate tensor per k-mer ----
        self.kmer_tensors = []    # list of (N, latent_dim) tensors
        self.kmer_missing = []    # list of (N,) bool tensors

        for path in [mer2_path, mer4_path, mer8_path]:
            if path is not None:
                vectors, missing = self._load_mer(path, sequences)
                self.kmer_tensors.append(vectors)
                self.kmer_missing.append(missing)

    def _load_mer(self, path, sequences):
        data = read_tsv(path)
        mer_dict = {row[0].lower(): [float(x) for x in row[4:]] for row in data[1:]}
        dim = len(next(iter(mer_dict.values())))

        rows = []
        missing = []
        for seq in sequences:
            seq_lower = seq.lower()
            if seq_lower in mer_dict:
                rows.append(mer_dict[seq_lower])
                missing.append(False)
            else:
                rows.append([0.0] * dim)
                missing.append(True)

        return (
            torch.tensor(rows, dtype=torch.float),
            torch.tensor(missing, dtype=torch.bool),
        )

    def _encode_all(self, sequence_list):
        encoded = []
        for seq in sequence_list:
            ids = [vocab.get(c, vocab["X"]) for c in seq[:self.max_seq_len]]
            pad_len = self.max_seq_len - len(ids)
            if pad_len > 0:
                ids = ids + [self.pad_token_id] * pad_len
            encoded.append(ids)
        return torch.tensor(encoded, dtype=torch.long)

    def _encode_labels(self, sequences, sequence_dict, label_idx, label_type):
        labels = []
        for seq in sequences:
            raw_label = sequence_dict[seq][label_idx]
            encoded = LABEL_MAPPINGS[label_type].get(raw_label, self.ignore_index)
            labels.append(encoded)
        return torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return self.all_tokens.size(0)

    def __getitem__(self, idx):
        # Always returns: tokens, mask, order, superfamily
        # Then for each k-mer: vector, missing_flag
        # With 3 k-mers: 4 + 6 = 10 elements total
        item = (
            self.all_tokens[idx],              # (L,)
            self.all_padding_masks[idx],       # (L,)
            self.all_order_labels[idx],        # scalar
            self.all_superfamily_labels[idx],  # scalar
        )
        for kmer_tensor, kmer_missing in zip(self.kmer_tensors, self.kmer_missing):
            item = item + (kmer_tensor[idx], kmer_missing[idx])
        return item































class SequenceDataset_with_topology_missingflagged(Dataset):
    def __init__(
        self,
        sequence_dict,
        max_seq_len,
        pad_token_id=0,
        ignore_index=-100,
        mer2_path=None,
        mer4_path=None,
        mer8_path=None,
        missing_lookup=None,   # {2: {seq_lower:True,...}, 4: {...}, 8: {...}}
    ):
        assert ignore_index is not None
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index
        self.missing_lookup = missing_lookup

        sequences = list(sequence_dict.keys())

        # ---- Tokens & Labels ----
        self.all_tokens = self._encode_all(sequences)
        self.all_padding_masks = (self.all_tokens == self.pad_token_id)
        self.all_order_labels = self._encode_labels(sequences, sequence_dict, 0, "Order")
        self.all_superfamily_labels = self._encode_labels(sequences, sequence_dict, 1, "Superfamily")

        # ---- K-mer latents + missing masks ----
        self.kmer_tensors = []   # list of (N, latent_dim)
        self.kmer_missing = []   # list of (N,) bool — True if missing

        for path, mer in [(mer2_path, 2), (mer4_path, 4), (mer8_path, 8)]:
            if path is not None:
                vectors, missing = self._load_mer(path, sequences, mer=mer)
                self.kmer_tensors.append(vectors)
                self.kmer_missing.append(missing)

    def _load_mer(self, path, sequences, mer=None):
        
        # MnTEdb
        image_files = glob.glob(f"./{mntedb_path}/*") #CHANGE PATH
        print(image_files)
        for file in image_files:
            with tarfile.open(file, "r:gz") as tar:
                all_files = tar.getnames()
                pkl_path = next((f for f in all_files if f.endswith('.pkl')), None)
                if pkl_path:
                    member = tar.getmember(pkl_path)
                    f = tar.extractfile(member)
                    data = pickle.load(f)
                    all_pkl = all_pkl | data

        return (
            torch.tensor(rows, dtype=torch.float),
            torch.tensor(missing, dtype=torch.bool),
        )

    def _encode_all(self, sequence_list):
        encoded = []
        for seq in sequence_list:
            ids = [vocab.get(c, vocab["X"]) for c in seq[: self.max_seq_len]]
            pad_len = self.max_seq_len - len(ids)
            if pad_len > 0:
                ids = ids + [self.pad_token_id] * pad_len
            encoded.append(ids)
        return torch.tensor(encoded, dtype=torch.long)

    def _encode_labels(self, sequences, sequence_dict, label_idx, label_type):
        labels = []
        for seq in sequences:
            raw_label = sequence_dict[seq][label_idx]
            encoded = LABEL_MAPPINGS[label_type].get(raw_label, self.ignore_index)
            labels.append(encoded)
        return torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return self.all_tokens.size(0)

    def __getitem__(self, idx):
        item = (
            self.all_tokens[idx],
            self.all_padding_masks[idx],
            self.all_order_labels[idx],
            self.all_superfamily_labels[idx],
        )
        for kmer_tensor, kmer_missing in zip(self.kmer_tensors, self.kmer_missing):
            item = item + (kmer_tensor[idx], kmer_missing[idx])
        return item


















class SequenceDataset_with_topology_missingflagged_older(Dataset):
    def __init__(
        self,
        sequence_dict,
        max_seq_len,
        pad_token_id=0,
        ignore_index=-100,
        mer2_path=None,
        mer4_path=None,
        mer8_path=None,
    ):
        assert ignore_index is not None
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index

        sequences = list(sequence_dict.keys())

        # ---- Tokens & Labels ----
        self.all_tokens = self._encode_all(sequences)
        self.all_padding_masks = (self.all_tokens == self.pad_token_id)
        self.all_order_labels = self._encode_labels(sequences, sequence_dict, 0, 'Order')
        self.all_superfamily_labels = self._encode_labels(sequences, sequence_dict, 1, 'Superfamily')

        # ---- K-mer latents + missing masks ----
        self.kmer_tensors = []    # list of (N, latent_dim)
        self.kmer_missing = []    # list of (N,) bool — True if missing

        for path in [mer2_path, mer4_path, mer8_path]:
            if path is not None:
                vectors, missing = self._load_mer(path, sequences)
                self.kmer_tensors.append(vectors)
                self.kmer_missing.append(missing)

    def _load_mer(self, path, sequences):
        """Load a mer TSV → (N, latent_dim) tensor + (N,) missing mask."""
        data = read_tsv(path)
        mer_dict = {row[0].lower(): [float(x) for x in row[4:]] for row in data[1:]}
        dim = len(next(iter(mer_dict.values())))

        rows = []
        missing = []
        for seq in sequences:
            seq_lower = seq.lower()
            if seq_lower in mer_dict:
                rows.append(mer_dict[seq_lower])
                missing.append(False)
            else:
                rows.append([0.0] * dim)
                missing.append(True)    # ← flag this sequence as missing

        return (
            torch.tensor(rows, dtype=torch.float),     # (N, latent_dim)
            torch.tensor(missing, dtype=torch.bool),   # (N,) True = missing
        )

    def _encode_all(self, sequence_list):
        encoded = []
        for seq in sequence_list:
            ids = [vocab.get(c, vocab["X"]) for c in seq[:self.max_seq_len]]
            pad_len = self.max_seq_len - len(ids)
            if pad_len > 0:
                ids = ids + [self.pad_token_id] * pad_len
            encoded.append(ids)
        return torch.tensor(encoded, dtype=torch.long)

    def _encode_labels(self, sequences, sequence_dict, label_idx, label_type):
        labels = []
        for seq in sequences:
            raw_label = sequence_dict[seq][label_idx]
            encoded = LABEL_MAPPINGS[label_type].get(raw_label, self.ignore_index)
            labels.append(encoded)
        return torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return self.all_tokens.size(0)

    def __getitem__(self, idx):
        # (tokens, mask, order, sf, mer2_vec, mer2_missing, mer4_vec, mer4_missing, ...)
        item = (
            self.all_tokens[idx],
            self.all_padding_masks[idx],
            self.all_order_labels[idx],
            self.all_superfamily_labels[idx],
        )
        for kmer_tensor, kmer_missing in zip(self.kmer_tensors, self.kmer_missing):
            item = item + (kmer_tensor[idx], kmer_missing[idx])
        return item




























class SequenceDataset_with_topology(Dataset):
    def __init__(
        self,
        sequence_dict,
        max_seq_len,
        pad_token_id=0,
        ignore_index=-100,
        mer2_path=None,
        mer4_path=None,
        mer8_path=None,
    ):
        assert ignore_index is not None

        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index

        sequences = list(sequence_dict.keys())

        # ---- Pre-encode tokens & labels (unchanged) ----
        self.all_tokens = self._encode_all(sequences)
        self.all_padding_masks = (self.all_tokens == self.pad_token_id)
        self.all_order_labels = self._encode_labels(sequences, sequence_dict, 0, 'Order')
        self.all_superfamily_labels = self._encode_labels(sequences, sequence_dict, 1, 'Superfamily')

        # ---- Pre-encode topology latents ----
        # Each mer file → dict of {seq_lower: 1D tensor}
        # We stack all available mers into (N, num_mers, latent_dim)
        mer_dicts = []
        mer_dims = []
        for path in [mer2_path, mer4_path, mer8_path]:
            if path is not None:
                data = read_tsv(path)
                d = {row[0].lower(): [float(x) for x in row[3:]] for row in data[1:]}
                dim = len(next(iter(d.values())))
                mer_dicts.append(d)
                mer_dims.append(dim)

        if mer_dicts:
            # Verify all mer dimensions match (they must for stacking)
            assert len(set(mer_dims)) == 1, \
                f"All mer latent dims must match, got {mer_dims}"
            self.latent_dim = mer_dims[0]
            self.num_mers = len(mer_dicts)

            # Build (N, num_mers, latent_dim) tensor
            all_topology = []
            for seq in sequences:
                seq_lower = seq.lower()
                mer_vectors = []
                for d in mer_dicts:
                    vec = d.get(seq_lower, [0.0] * self.latent_dim)
                    mer_vectors.append(vec)
                all_topology.append(mer_vectors)

            # (N, num_mers, latent_dim)
            self.all_topology = torch.tensor(all_topology, dtype=torch.float)
        else:
            self.all_topology = None

    def _encode_all(self, sequence_list):
        encoded = []
        for seq in sequence_list:
            ids = [vocab.get(c, vocab["X"]) for c in seq[:self.max_seq_len]]
            pad_len = self.max_seq_len - len(ids)
            if pad_len > 0:
                ids = ids + [self.pad_token_id] * pad_len
            encoded.append(ids)
        return torch.tensor(encoded, dtype=torch.long)

    def _encode_labels(self, sequences, sequence_dict, label_idx, label_type):
        labels = []
        for seq in sequences:
            raw_label = sequence_dict[seq][label_idx]
            encoded = LABEL_MAPPINGS[label_type].get(raw_label, self.ignore_index)
            labels.append(encoded)
        return torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return self.all_tokens.size(0)

    def __getitem__(self, idx):
        # Returns: (tokens, padding_mask, order_label, superfamily_label, topology)
        item = (
            self.all_tokens[idx],              # (L,)
            self.all_padding_masks[idx],       # (L,)
            self.all_order_labels[idx],        # scalar
            self.all_superfamily_labels[idx],  # scalar
        )
        if self.all_topology is not None:
            # (num_mers, latent_dim) — e.g. (3, 128)
            return item + (self.all_topology[idx],)
        else:
            return item + (None,)
        







# condor_submit mntedb.sub
# condor_submit mntedb_sf.sub

# condor_submit repetdb.sub
# condor_submit repetdb_sf.sub

# condor_submit repbase.sub
# condor_submit repbase_sf.sub













class PersistenceDataset(Dataset):
    def __init__(self, data_dict, datasets):
        self.x4 = data_dict[datasets[0]]
        self.x8 = data_dict[datasets[1]]
        self.x14 = data_dict[datasets[2]]
        self.x20 = data_dict[datasets[3]]
        self.label = data_dict['Label']
        self.label_id = data_dict['label_id']
        self.seq_x = data_dict['seq_x']
        self.dataset = data_dict['dataset']
    def __len__(self):
        return len(self.x4)
    def __getitem__(self, idx):
        # Adjust input size to [Batch, 5, 128, 128]
        x4 = torch.tensor(self.x4[idx], dtype=torch.float32)
        x4 = x4.permute(2,0,1)
        x8 = torch.tensor(self.x8[idx], dtype=torch.float32)
        x8 = x8.permute(2,0,1)
        x14 = torch.tensor(self.x14[idx], dtype=torch.float32)
        x14 = x14.permute(2,0,1)
        x20 = torch.tensor(self.x20[idx], dtype=torch.float32)
        x20 = x20.permute(2,0,1)
        # Labels (assuming they are already numerical or encoded)
        seq_x = self.seq_x[idx]
        label = self.label[idx]
        label_id = self.label_id[idx]
        dataset = self.dataset[idx]
        # return x4, x8, x14, seq_x, label_id, label, dataset
        return x4, x8, x14, x20, seq_x, label_id, label, dataset



def get_PI(path):
    all_pkl = {}
    image_files = glob.glob(f"./{path}/*")
    # print(image_files)
    for file in image_files:
        with tarfile.open(file, "r:gz") as tar:
            all_files = tar.getnames()
            pkl_path = next((f for f in all_files if f.endswith('.pkl')), None)
            if pkl_path:
                member = tar.getmember(pkl_path)
                f = tar.extractfile(member)
                data = pickle.load(f)
                all_pkl = all_pkl | data
    return all_pkl

def update_metadata(lookup, all_pkl):
    for seq, metadata in lookup.items():
        if seq in all_pkl:
            all_pkl[seq].update(metadata)
        else:
            all_pkl[seq] = {
                **metadata,
                'persistence_image': np.zeros((128,128,5)) #add 0 if none
            }
    return all_pkl


def split_data(all_pkl, lookup):
    filtered_pkl = {}

    for seq, metadata in lookup.items():
        if seq in all_pkl:
            # Create a shallow copy so the original all_pkl stays untouched
            entry = all_pkl[seq].copy()
            entry.update(metadata)
            filtered_pkl[seq] = entry
        else:
            # Create a brand new entry for sequences not in all_pkl
            filtered_pkl[seq] = {
                **metadata,
                'persistence_image': np.zeros((128, 128, 5))
            }
    return filtered_pkl

def prep_for_dataloader(all_pkl, n):
    # Get data ready for dataloader
    train = []
    label = []
    label_id = []
    sequences = []
    dataset = []

    for key, values in all_pkl.items():
        train.append(values['persistence_image'])
        label.append(values['labels'])
        label_id.append(values['label_id'])
        sequences.append(key)
        dataset.append(values['dataset'])

    dataset = {f'x{n}': train, 'label_id':label_id, 'Label':label, 'seq_x':sequences, 'dataset':dataset}
    return dataset

def get_train_test(all_pkl, train_lookup, test_lookup):
    train = split_data(all_pkl, train_lookup)
    test = split_data(all_pkl, test_lookup)
    return train, test

def dataloader_ready(train, test, n):
    train = prep_for_dataloader(train, n)
    test = prep_for_dataloader(test, n)
    return train, test

def get_test_train_split(all_pkl, labels, i):
    train = labels[labels[f'fold_{i}'] == 'train']
    test = labels[labels[f'fold_{i}'] == 'test']

    train_lookup = train.drop_duplicates('seq_x').set_index('seq_x')[['label_id', 'labels',f'fold_{i}', 'dataset']].to_dict('index')
    test_lookup = test.drop_duplicates('seq_x').set_index('seq_x')[['label_id', 'labels',f'fold_{i}', 'dataset']].to_dict('index')

    train, test = get_train_test(all_pkl, train_lookup, test_lookup)

    train, test = dataloader_ready(train, test, i)

    return train, test



# Create tree and id mappings
ORDER_TO_SUPERFAMILIES={'DIRS': [],
 'Helitron': [],
 'Line': ['CR1', 'I', 'Jockey', 'L1', 'R2', 'RTE', 'Rex1'],
 'LTR': ['Bel-Pao', 'Copia', 'Gypsy', 'ERV'],
 'PLE': [],
 'Sine': ['SINE', 'SINE1/7SL', 'SINE2/tRNA', 'SINE3/5S'],
 'TIR': ['CACTA', 'MuLE', 'PIF', 'TcMar', 'hAT']}

# Build tree
root = build_classification_tree(ORDER_TO_SUPERFAMILIES)
label_map = build_label_to_node_id(root)

# Load labels
labels = pd.read_csv(labels_path, sep='\t')
labels['label_id'] = labels['labels'].map(label_map)
lookup = labels.drop_duplicates('seq_x').set_index('seq_x')[['label_id', 'labels', 'dataset']].to_dict('index')

pi_paths = [f'{pi_dir}/4mer', f'{pi_dir}/8mer', f'{pi_dir}/14mer', f'{pi_dir}/20mer']

all_pkl = {}
train_images = []

for p in pi_paths:
    pkl_file = get_PI(p)
    pkl_file = update_metadata(lookup, pkl_file)
    
    folds = {}
    for i in range(5):
        train, test = get_test_train_split(pkl_file, labels, i=i)
        folds[f'split_{i}'] = {'train': train, 'test': test}

    all_pkl[p] = folds # for given database, all the splits for all kmers

datasets = list(all_pkl.keys())
train_dict = {}
test_dict = {}

all_results = {} # To save all results

torch.manual_seed(777)
for fold in range(5):
    print(f"fold {fold}")

    model = HierarchicalCNN(root_node=root).to(device)
    criterion = HierarchicalSoftmaxLoss(root=root)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train_dict = {}
    test_dict = {}
    split = f'split_{fold}'

    for key, value in all_pkl.items():
        
        train = value[split]['train'][f'x{fold}']
        train_label = value[split]['train']['Label']
        train_id = value[split]['train']['label_id']
        train_seq = value[split]['train']['seq_x']
        train_ds = value[split]['train']['dataset']
        
        test = value[split]['test'][f'x{fold}']
        test_label = value[split]['test']['Label']
        test_id = value[split]['test']['label_id']
        test_seq = value[split]['test']['seq_x']
        test_ds = value[split]['test']['dataset']

        train_dict.update({'Label': train_label, 'label_id': train_id, 'seq_x': train_seq, 'dataset': train_ds, f'{key}': train})
        test_dict.update({'Label': test_label, 'label_id': test_id, 'seq_x': test_seq, 'dataset': test_ds, f'{key}': test})
    
    train_dataset = PersistenceDataset(train_dict, datasets)
    test_dataset = PersistenceDataset(test_dict, datasets)