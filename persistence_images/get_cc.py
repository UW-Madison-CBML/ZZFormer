import pandas as pd
import tarfile
import numpy as np
import pickle
import io
import itertools
from collections import Counter, defaultdict
import sys
import glob
# from gtda.diagrams import PersistenceImage
# from gtda.plotting import plot_heatmap
import dionysus
import numpy as np
from scipy.stats import multivariate_normal


def get_nuc_counts(nodes):
    results = []
    for t in nodes:
        combined_string = "".join(t)
        counts = dict(Counter(combined_string))
        results.append(counts)
    return results

def set_atgcx_5(all_node_counts_list):
    rgb_atgcx = []
    for data in all_node_counts_list:
        if len(data) > 0:
            a_val = data.get('a', 0)
            t_val = data.get('t', 0)
            g_val = data.get('g', 0)
            c_val = data.get('c', 0)
            x_val = data.get('x', 0)

            total = sum(data.values())
            a = (a_val / total) #* 255
            t = (t_val / total) #* 255
            g = (g_val / total) #* 255
            c = (c_val / total) #* 255
            x = (x_val / total) #* 255

            rgb_atgcx.append((a,t,g,c,x))
        else:
            rgb_atgcx.append((0,0,0,0,0))
    return rgb_atgcx

def set_counts(all_node_counts_list):
    rgb_atgcx = []
    for data in all_node_counts_list:
        
        a_val = data.get('a', 0)
        t_val = data.get('t', 0)
        g_val = data.get('g', 0)
        c_val = data.get('c', 0)
        x_val = data.get('x', 0)

        total = sum(data.values())
        a = (a_val) # / total) #* 255
        t = (t_val) # / total) #* 255
        g = (g_val) # / total) #* 255
        c = (c_val) # / total) #* 255
        x = (x_val)
        rgb_atgcx.append((a,t,g,c,x))
    return rgb_atgcx


def get_alphabet(a:list, k:int):
    alphabet = a * k
    combos = [''.join(c) for c in itertools.combinations(alphabet, k)]
    sorted_combos = sorted(set(combos)) 
    node_id = {node: i for i, node in enumerate(sorted_combos)} 

    return sorted_combos, node_id


def process_barcodes(obj):
    all_barcodes = []
    labels = []
    all_nodes = []
    all_node_counts = []
    for seq, value in obj.items():
        l = len(seq)
        for k, bd in value[0].items():

            if len(bd[-1]) == 0:
                b_d_times = []
                continue
            else:
                b_d_times = []
                for birth, death in bd[-1]:
                    if death == np.inf:
                        death = l
                    b_d_times.append([int(birth), int(death)-int(birth)]) # Calculate persistence
                labels.append(seq)
                all_barcodes.append(np.array(b_d_times))
            nodes = [tuple(id_to_node[num] for num in t) for t in bd[0]]
            all_nodes.append(nodes)

            node_counts = get_nuc_counts(nodes)
            all_node_counts.append(node_counts)

    return all_nodes, all_node_counts, all_barcodes, labels

def process_tarball(path, filename, target_dict_name):
    all_data = []
    labels = []
    # Open the tar.gz
    with tarfile.open(f"{path}/{filename}", "r:gz") as tar:
        for member in tar:

            if target_dict_name in member.name and member.isfile():
                pkl_file = tar.extractfile(member)

                if pkl_file:
                    obj = pickle.load(pkl_file)

                    # Get barcode and sequence
                    all_nodes, all_node_counts, all_barcodes, l = process_barcodes(obj)
                    all_data.append(all_barcodes)
                    labels.append(l)
    return all_data, labels, all_nodes, all_node_counts

def get_grid_and_weights(diagrams, n_bins=128, weight_func=None):

    all_births = np.concatenate([d[:, 0] for d in diagrams])
    all_pers = np.concatenate([d[:, 1] for d in diagrams])
    
    b_min, b_max = all_births.min(), all_births.max()
    p_min, p_max = all_pers.min(), all_pers.max()
    
    b_sampling = np.linspace(b_min, b_max, n_bins)
    p_sampling = np.linspace(p_min, p_max, n_bins)
    
    if weight_func is None:
        weights = np.ones(n_bins)
    else:
        weights = weight_func(p_sampling)
        
    return (b_sampling, p_sampling), weights

def generate_images_global_intensity(diagrams, sampling, weights, arr, sigma=0.5):
    b_samples, p_samples = sampling
    n_bins = len(b_samples)
    X, Y = np.meshgrid(b_samples, p_samples)
    grid_coords = np.dstack([X, Y])
    
    images = []
    
    for dgm in diagrams:

        # global_intensities = np.sum(dgm_channels, axis=0) / 255.0
        
        density_map = np.zeros((n_bins, n_bins))
        
        coords = np.vstack([dgm[:, 0], dgm[:, 1]]).T
        
        for i, point in enumerate(coords):
            rv = multivariate_normal(point, [[sigma, 0], [0, sigma]])
            pdf_slice = rv.pdf(grid_coords)
            
            p_idx = np.abs(p_samples - point[1]).argmin()
            w = weights[p_idx]
        
            density_map += (w * pdf_slice)
            
    img_5ch = density_map[:, :, np.newaxis] * arr
        
    images.append(img_5ch)
        
    return np.array(images)


path = sys.argv[1]
filename = sys.argv[2]

target_dict_name = 'results'

all_data = []
labels = []
# Open the tar.gz
with tarfile.open(f"{path}/{filename}", "r:gz") as tar:
    for member in tar:

        if target_dict_name in member.name and member.isfile():
            pkl_file = tar.extractfile(member)

            if pkl_file:
                obj = pickle.load(pkl_file)
                
                # Get barcode and sequence
                # all_nodes, all_node_counts, all_barcodes, l = process_barcodes(obj)
                all_barcodes = []
                labels = []
                # all_nodes = [] # equivalent of cc
                all_node_counts = []
                for seq, value in obj.items():
                    l = len(seq)

                    barcodes = list(value[0]['barcode_H1'])
                    elements = list(value[0]['barcode_elements_H1'])
                    node_id = value[0]['node_id']
                    id_to_node = {v:k for k,v in node_id.items()}

                    ccs = [tuple(id_to_node[num] for num in t) for t in elements] # list of connected components
                    nodes = [node for edge in ccs for node in edge]
                    combined_string = "".join(nodes)
                    node_counts = dict(Counter(combined_string)) # counts of atgcx for given sequence
                    
                    b_d_times = []

                    # Iterate through each CC and its birth/death times
                    for bd, cc in zip(barcodes, ccs):
                        
                        if len(bd) == 0:
                            continue
                        else:
                            birth, death = bd
                            if death == np.inf:
                                death = l
                            b_d_times.append([int(birth), int(death)-int(birth)]) # Calculate persistence
                            
                    if len(b_d_times) > 0:
                        labels.append(seq)
                        all_barcodes.append(np.array(b_d_times))
                        all_data.append(node_counts)

rgb_atgcx = set_atgcx_5(all_data)
atgcx = set_counts(all_data)

persistence_images = {}

for b,s,arr,c in zip(all_barcodes, labels, rgb_atgcx, atgcx):
    diagrams = [b]

    sampling, weights = get_grid_and_weights(diagrams, n_bins=128, weight_func=None)
    image = generate_images_global_intensity(diagrams, sampling, weights, [arr], sigma=0.5)
    if image.max() > 0:
        image = image / image.max()
    
    persistence_images[s] = {'persistence_image': image.squeeze(), 'intensities': [arr], 'inten_counts':[c]}

save_name = filename.replace('.tar.gz', '')
with open(f"./{save_name}.pkl", "wb") as f:
    pickle.dump(persistence_images, f)
