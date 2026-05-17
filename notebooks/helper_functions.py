import dionysus as d
import math
import matplotlib.pyplot as plt
import numpy as np
import KK_zz_apex_LS as kkkk
import networkx as nx
from collections import defaultdict, Counter
import itertools
from itertools import combinations

# def get_alphabet(a:list, k:int):
#     """
#     Inputs: 
#         a: list of alphabets
#         k: k-mer

#     Outputs:
#         sorted_combos: sorted list of all nodes
#         node_id: dict mapping letter to id
#     """
#     alphabet = a * k

#     # Precompute node id mapping for entire sequence (global ids)
#     combos = [''.join(c) for c in itertools.combinations(alphabet, k)]
    
#     # Unique combinations of nodes
#     sorted_combos = sorted(set(combos)) 

#     # Asign node IDs
#     node_id = {node: i for i, node in enumerate(sorted_combos)} 

#     return sorted_combos, node_id

def get_alphabet(seq, k): # for larger kmers, only get the nodes that exist
    unique_kmers = set()
    for i in range(len(seq) - k + 1):
        kmer = seq[i : i + k]
        unique_kmers.add(kmer)
    sorted_combos = sorted(list(unique_kmers))
    node_id = {node: i for i, node in enumerate(sorted_combos)} 
    return sorted_combos, node_id

    
def get_edges(seq, k, node_id, l=1):
    """
    Inputs: 
        seq: total sequence
        k: k-mer
        node_id: dict mapping letter to id

    Outputs:
        edges: edges in id form
        nodes: nodes in id form
    """

    # Get nodes
    nodes = [seq[i:i+k] for i in range(len(seq)-k+1)]

    # Get the edges
    edges = []
    for i in range(len(nodes) - l):
        a = nodes[i]
        b = nodes[i + l]
        # if a == b:
        #     continue
        edge = tuple((a, b))
        edges.append(edge)

    # Convert to IDs
    edges = [(node_id[u], node_id[v]) for u, v in edges]
    nodes = [node_id[n] for n in nodes]

    return edges, nodes

def get_triangles(edges_at_t):
    """
    Function to find the existing triangles
    """

    # Find the triangles
    G = nx.Graph()
    G.add_edges_from(edges_at_t)
    triangles = [tuple(sorted(c)) for c in nx.enumerate_all_cliques(G) if len(c) == 3]

    return triangles

def traverse_sequence(edges, m):
    """
    Actual traversal of sequence, keeping track of memory
    Output:
        List of simplices (not including unions)
    """
    memory = defaultdict(int)

    simplices = []
    to_remove = []
    ####################################################################################
    # Sequence traversal
    for edge in edges:

        # If edge exists in memory, reset its memory
        # if edge in memory.keys():
        #     memory[edge] = m

        # If edge needs to be remove, remove
        for e in to_remove:
            del memory[e]

        # If edge not in memory, add it and initialize its memory
        if edge not in memory.keys():
            memory[edge] = m

        # Iterate through the memory dict
        for e, life in memory.items():
            # -1 from memory if the edge wasn't seen
            if e != edge:
                memory[e] -= 1
            # Edge already exists, renew its memory
            elif e == edge:
                memory[e] = m

        # Iterate through and find any edge that died to remove next iter
        to_remove = []
        for e, life in memory.items():
            if life == 0:
                to_remove.append(e)

        simplices.append(list(memory.keys()))

    return simplices


def unionize_timepoints(simplicies):
    """
    Calculate the unions
    """
    simplex_unions = []
    for i in range(len(simplicies)-1):
        simplex_unions.append(simplicies[i])
        union = list(set(simplicies[i]) | set(simplicies[i+1]))
        simplex_unions.append(union)

    if simplicies:
        simplex_unions.append(simplicies[-1])
    return simplex_unions
    

def prep_simplicies(unions):
    # Set times for the unions
    times_dict = defaultdict(list)
    # for i, simplices in enumerate(all_simplices):
    for t, simplex in enumerate(unions):
        for edge in simplex:
            times_dict[edge].append(t+1)

    # Remove the self edges
    self_edges = []
    for edge in times_dict.keys():
        if len(edge) == 2:
            vi, vj = edge
            if vi == vj:
                self_edges.append(edge)

    for edge in self_edges:
        del times_dict[edge]

    return times_dict, self_edges


def get_birth_death_from_time_list(times_dict):
    """
    Get the birth and death times
    """
    bd = {}
    for key, times in times_dict.items():
        if not times:
            bd[key] = []
            continue
        t = sorted(set(times))
        result = []
        run_start = t[0]
        for prev, curr in zip(t, t[1:]):
            if curr == prev + 1:
                continue
            else:
                result.append(run_start)
                result.append(prev+1)  # exclusive
                run_start = curr

        # finalize last run
        result.append(run_start)
        result.append(t[-1] + 1)

        bd[key] = result

    return bd

def prep_for_fastZZ(birth_death_times, node_id):
    """
    Outputs the list of simplices and birth,death times for dionysus fast ZZZ computation
    """
    # Assemble times and simplices for Dionysus
    times_final = []
    simplices_final = []

    for n in node_id.values():
        simplices_final.append([n])
        times_final.append([0])

    for e, t in birth_death_times.items():
        simplices_final.append(list(e))
        times_final.append(t)
    
    return simplices_final, times_final


def run_fastZZ(filtered_simplices,filtered_times):
    cone = d.fast_zigzag(filtered_simplices, filtered_times)
    r,v = d.homology_persistence(cone, method = 'matrix_v')
    dgms = d.init_zigzag_diagrams(r,cone)
    return cone, r, v, dgms

def GetZZDionysus_H1(cone, r, v, dgms):

    # barcodes_and_elements={}
    barcodes_elements=[]
    barcodes=[]
    for dim,type_dgm in enumerate(dgms):
        if dim==1:
        #if dim == dim:
            #print("Dimension:", dim)
            for typppe,dgm in type_dgm.items():
                for pt in dgm:
                    barcodes.append((float(pt.birth),float(pt.death)))
                    #print("Barcode time:",pt, pt.data)
                    apex_rep = kkkk.apex(pt,r,v,cone)
                    nodes = sorted({
                        v
                        for (_, (x, _)) in apex_rep
                        for v in cone[x]
                    })
                    nodes_tuple = tuple(nodes)
                    barcodes_elements.append(nodes_tuple)
                    
    return barcodes_elements, barcodes



def handle_H1_barcodes(barcode_elements, barcode, l, k):
    remove_idx = []

    for i, current_elements in enumerate(barcode_elements):
        if len(current_elements) <= 2:
            remove_idx.append(i)

    filt_elements = [x for i, x in enumerate(barcode_elements) if i not in remove_idx]
    filt_barcodes = [x for i, x in enumerate(barcode) if i not in remove_idx]

    H1_mappings = {k: i for i, k in enumerate(sorted(set(filt_elements)), start=1)}

    CC_dict = defaultdict(list)

    CC_vector = [0] * l
    pers_vals = [0] * l

    for CC, BC in zip(filt_elements, filt_barcodes):
        id = H1_mappings[CC]
        CC_dict[id].append(BC)

        for i in range(int(BC[0]), int(BC[1])):
            CC_vector[i] = id
            pers_vals[i] = (BC[1] - BC[0]) / l

    idx = (l-(k-1)) - 1

    # if CC_vector[idx] != 0 :
    #     CC_vector[idx:] = [CC_vector[idx]] * (k-1)
    #     pers_vals[idx:] = [pers_vals[idx]] * (k-1)

    return CC_vector, pers_vals, H1_mappings, filt_elements, filt_barcodes




















def plot_barcodes(simplices_final, times_final, max, kmer, dims=[0,1,2]):

    cone = d.fast_zigzag(simplices_final, times_final)
    r,v = d.homology_persistence(cone, method = 'matrix_v')
    dgms = d.init_zigzag_diagrams(r,cone)

    counter = 1
    plt.figure()
    for dim in dims:
        if dim == 0:
            c = 'blue'
        elif dim == 1:
            c = 'red'
        elif dim == 2:
            c = 'green'

        for type in ['co', 'oc', 'oo', 'cc']:
            for pt in dgms[dim][type]:
                birth = pt.birth
                if pt.death == np.inf:
                    death = max
                else:
                    death = pt.death
                plt.hlines(counter, birth, death, colors=c)
                counter += 1
    plt.title(kmer)
    plt.show()


def plot_diagrams(simplices_final, times_final, max, kmer, save, dims=[0,1,2]):

    cone = d.fast_zigzag(simplices_final, times_final)
    r,v = d.homology_persistence(cone, method = 'matrix_v')
    dgms = d.init_zigzag_diagrams(r,cone)

    plt.figure()
    plt.plot([0, max+1], [0, max+1])
    for dim in dims:
        if dim == 0:
            c = 'blue'
        elif dim == 1:
            c = 'red'
        elif dim == 2:
            c = 'green'

        for type in ['co', 'oc', 'oo', 'cc']:
            for pt in dgms[dim][type]:
                birth = pt.birth
                if pt.death == np.inf:
                    death = max
                else:
                    death = pt.death
                plt.scatter(birth, death, color=c, s=1)
    plt.title(kmer)
    #plt.savefig(save)
    plt.show()


def get_perc_values(barcode_elements, barcode, l):
    barcode_dict = {}
    
    for i, bc in enumerate(barcode_elements):
        b, d = barcode[i]

        barcode_dict[i] = (d-b) / l

        return barcode_dict
