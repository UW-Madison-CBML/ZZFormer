import os
import tarfile
import pickle
import glob
import io
import sys

os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/torch_cache"
os.environ["USER"] = "researcher"
os.environ["LOGNAME"] = "researcher"

from hierarchicalsoftmax import SoftmaxNode, HierarchicalSoftmaxLoss, HierarchicalSoftmaxLinear
from hierarchicalsoftmax.inference import (
    greedy_predictions,
    node_probabilities,
)

import numpy as np
from collections import defaultdict, Counter
import itertools
import pandas as pd

# import dionysus

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from sklearn.model_selection import StratifiedKFold

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

# class Encoder(nn.Module):
#     def __init__(
#             self, 
#             n_channels=5,
#             n_filters=16,):
#         super(Encoder, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(n_channels, n_filters, kernel_size=3, padding=1), 
#             nn.ReLU(),
#             nn.MaxPool2d(2),        # Output: [16, 64, 64]
#             nn.Conv2d(n_filters, n_filters//2, kernel_size=3, padding=1), 
#             nn.ReLU(),
#             nn.MaxPool2d(2),        # Output: [8, 32, 32]
#             nn.Flatten()            # Output: [8192]
#         )
#     def forward(self, x):
#         return self.encoder(x)

class Encoder(nn.Module):
    def __init__(self, n_channels=5, n_filters=16):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            # 128 to 64
            nn.Conv2d(n_channels, n_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # 64 to 32
            nn.Conv2d(n_filters, n_filters*2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # 32 to 16
            nn.Conv2d(n_filters*2, n_filters*4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Average Pooling: set wanted output size, not kernel
            nn.AdaptiveAvgPool2d((1, 1)), # (64, 1, 1)
            nn.Flatten()                  # (64)
        )

    def forward(self, x):
        return self.encoder(x)

class HierarchicalCNN(nn.Module):
    def __init__(self, root_node, n_channels=5, n_filters=16):
        super(HierarchicalCNN, self).__init__()
        
        # These now return a vector of size 64 each (if n_filters=16)
        self.enc4 = Encoder(n_channels=n_channels, n_filters=n_filters)
        self.enc8 = Encoder(n_channels=n_channels, n_filters=n_filters)
        self.enc14 = Encoder(n_channels=n_channels, n_filters=n_filters)
        self.enc20 = Encoder(n_channels=n_channels, n_filters=n_filters)
        
        # Calculate the concatenated size: (n_filters * 4) * 4 encoders
        combined_size = (n_filters * 4) * 4 
        
        self.prediction_head = nn.Sequential(
            nn.Linear(combined_size, 512), # 256 to 512
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )
        
        self.hierarchical_layer = HierarchicalSoftmaxLinear(
            in_features=512,
            root=root_node
        )

    def forward(self, x4, x8, x14, x20):
        # Your forward logic remains exactly the same!
        z4 = self.enc4(x4)
        z8 = self.enc8(x8)
        z14 = self.enc14(x14)
        z20 = self.enc20(x20)
        
        combined = torch.cat((z4, z8, z14, z20), dim=1)
        classification = self.prediction_head(combined)
        return self.hierarchical_layer(classification)


def build_classification_tree(
    order_to_superfamilies: dict,
    label_smoothing: float = 0.0,
    gamma: float = 0.0,
) -> SoftmaxNode:
    """
    Builds a 2-level hierarchical softmax tree.

    Args:
        order_to_superfamilies: e.g. {
            "LINE": ["CR1", "L1", "L2", "Jockey", "RTE"],
            "SINE": ["Alu", "MIR", "tRNA"],
            "DNA":  ["hAT", "TcMar", "Merlin"],
            ...
        }
    Returns:
        root: The root SoftmaxNode with set_indexes() already called.
    """
    root = SoftmaxNode(
        "root",
        label_smoothing=label_smoothing,
        gamma=gamma,
    )
    for order_name, superfamily_list in order_to_superfamilies.items():
        order_node = SoftmaxNode(
            order_name,
            parent=root,
            label_smoothing=label_smoothing,
            gamma=gamma,
        )
        for sf_name in superfamily_list:
            SoftmaxNode(
                sf_name,
                parent=order_node,
                label_smoothing=label_smoothing,
                gamma=gamma,
            )
    root.set_indexes()
    return root

def build_label_to_node_id(root: SoftmaxNode) -> dict:
    """
    Builds a mapping from node name strings → node_id integers.
    These integer IDs are what you pass as target_node_ids during training.
    They index into root.node_list, which is what HierarchicalSoftmaxLoss uses.
    """
    root.set_indexes_if_unset()
    label_to_id = {}
    for node_id, node in enumerate(root.node_list):
        # Full path (e.g., "LINE/CR1")
        if node.parent and not node.parent.is_root:
            full_name = "/".join(
                [str(n) for n in node.ancestors[1:]] + [str(node)]
            )
        else:
            full_name = str(node)
        label_to_id[full_name] = node_id
        # Short name if unambiguous
        short_name = str(node)
        if short_name not in label_to_id:
            label_to_id[short_name] = node_id
    return label_to_id


def node_lineage_string(node) -> str:
    """Convert a SoftmaxNode to its full lineage path string."""
    if node.is_root:
        return "Unknown"
    return "/".join([str(n) for n in node.ancestors[1:]] + [str(node)])

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

labels_path = sys.argv[1] #'hsm_MnTEdb_labels.tsv'
pi_dir = sys.argv[2] #'MnTEdb/'
save_name = sys.argv[3] #'MnTEdb'

device = torch.device('cuda')
n_epochs = 100

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

    torch.manual_seed(777)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    torch.cuda.empty_cache()
    for epoch in range(n_epochs):
        model.train()
        for x4, x8, x14, x20, seq, ids, label, d in train_loader:
            x4 = x4.to(device)
            x8 = x8.to(device)
            x14 = x14.to(device)
            x20 = x20.to(device)
            ids = ids.to(device)
            # Forward pass 
            outputs = model(x4, x8, x14, x20)
            # Loss
            loss = criterion(outputs, ids)
            # Backwards pass
            optimizer.zero_grad() # Reset gradients from last batch
            loss.backward() # Backpropagation
            optimizer.step()      # Update parameters
            # running_loss += loss.item()
        print(f"\nTraining loss:  {loss:.4f}")

    save_path = f'/staging/svaren/zzformer/{save_name}_f{fold}.pth'
    torch.save({
        'epoch': n_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        # 'loss': loss
    }, save_path)

    # INFERENCE
    preds = []
    trues = []
    outss = []

    model.eval() 
    with torch.no_grad():
        for x4, x8, x14, x20, seq, ids, label, d in test_loader:
            x4 = x4.to(device)
            x8 = x8.to(device)
            x14 = x14.to(device)
            x20 = x20.to(device)

            outputs = model(x4, x8, x14, x20)
            predictions = greedy_predictions(outputs, root=root)

            outss.extend(outputs)
            preds.extend(predictions)
            trues.extend(label)

    all_results[f"fold_{fold}"] = {'pred_y': preds,
                                   'test_y': trues,
                                   'outs': outss,
                                   'fold': fold}

with open(f'/staging/svaren/zzformer/{save_name}_all_results.pkl', 'wb') as f:
    pickle.dump(all_results, f)
