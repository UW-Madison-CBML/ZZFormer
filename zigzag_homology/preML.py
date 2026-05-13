import dionysus as d
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
import numpy as np

import networkx as nx
from collections import defaultdict, Counter
import itertools
from itertools import combinations

from helper_functions import *
from KK_zz_apex_LS import *
# from zznetvis_helper_functions_minimal_LS import *
# from plot_graph_LS import PlotGraphs3

from datasets import load_dataset, Features, Value
import pickle 
from collections import defaultdict
import sys
import json

dataset = sys.argv[1]

file = f'{dataset}.txt'

with open(file) as f:
    sequences = [line.strip() for line in f]

count = sum(
    1
    for seq in sequences
    if any(l.lower() not in "atgcx" for l in seq)
)
if count != 0:
    print(f"NON APPROVED LETTERS PRESENT IN SEQUENCES OF {dataset}")
    sys.exit(1)


############## Run zz homology ##############
a = ['a', 'c', 't', 'g', 'x']
l = 1

k = 4
memory = 100

counter = 0
all_results = defaultdict(list)

for seq in sequences:
    
    l = len(seq)
    print(f"Seq: {counter} Length: {l}")

    # TDA
    sorted_combos, node_id = get_alphabet(seq, k)
    edges, nodes = get_edges(seq, k, node_id)

    simplicies = traverse_sequence(edges, memory)
    unionized = unionize_timepoints(simplicies)

    times_dict, self_edges = prep_simplicies(unionized)
    birth_death_times = get_birth_death_from_time_list(times_dict)
    simplices_final, times_final = prep_for_fastZZ(birth_death_times, node_id)
    cone, r, v, dgms = run_fastZZ(simplices_final,times_final)

    barcode_elements_H1, barcode_H1 = GetZZDionysus_H1(cone, r, v, dgms)

    all_results[seq].append({'barcode_elements_H1':barcode_elements_H1, 'barcode_H1':barcode_H1, 'node_id':node_id})

    counter += 1



############## Save results in pkl ##############
with open(f'results_{dataset}_{k}mer.pkl', 'wb') as f:
    pickle.dump(all_results, f)

