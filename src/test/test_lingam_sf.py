#!/usr/bin/env python3

from causallearn.utils.cit           import fisherz      # ∼O(n³) linear-Gaussian CI test
from causallearn.utils.GraphUtils    import GraphUtils
import dowhy.datasets as ds          # DoWhy ≥ 0.10


import numpy as np
import time

import sys, pathlib
repo_root = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(repo_root / "src"))


from causaldiscovery.algorithms.LiNGAM_SF import lingam_sf
from causaldiscovery.CItest.noCache_CI_Test import myTest
from causaldiscovery.utils.auxiliary import make_bn_truth_and_sample




import importlib.util, sys

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from cdt.data import AcyclicGraphGenerator

# BASE RANDOM GENERATOR TO GUARANTEE REPRODUCIBILITY
base_rng = np.random.default_rng(123)

def draw_digraph(G, filename=None):
    # choose a layout
    pos = None
    if nx.is_directed_acyclic_graph(G):
        # Prefer Graphviz 'dot' (nice top→bottom DAG layout) if available
        try:
            from networkx.drawing.nx_pydot import graphviz_layout
            pos = graphviz_layout(G, prog="dot")
        except Exception:
            pos = nx.dag_layout(G)  # fallback DAG layout
    if pos is None:
        # generic fallback for non-DAGs or if Graphviz isn't installed
        k = 1 / np.sqrt(max(1, G.number_of_nodes()))
        pos = nx.spring_layout(G, seed=0, k=k)

    # light-weight styling that works for ~100 nodes
    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(G, pos, node_size=40, alpha=0.9)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle='-|>', arrowsize=10, width=0.6, alpha=0.7)
    # labels are optional for big graphs; uncomment if you want them
    # nx.draw_networkx_labels(G, pos, font_size=7)

    plt.axis('off')
    if filename:
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


digraph, df = make_bn_truth_and_sample(
    base_rng = base_rng ,
    num_vars = 50,
    expected_degree = 2,
    n_samples = 8000
)       

#draw_digraph(digraph, "Ground-Truth DAG")


data   = df.values                   # numpy array, n×p
names = df.columns


data1 = data[:, 0:10]
names1 = names[ 0:10]
names2 = names[ 10: ]

CI_test = myTest(df)

t0 = time.perf_counter()
mag, nCI, avg_sepset_size, total_exec_time = lingam_sf(
    data1,
    independence_test_method=CI_test,
    new_node_names=names1,
    alpha1 = 0.05,
    verbose=False,
    getMag = True
)
t1 = time.perf_counter()

print("Primeras 10 var")

mag, nCI, avg_sepset_size, total_exec_time = lingam_sf(
    data,
    independence_test_method=CI_test,
    initial_graph = mag,
    new_node_names=names2,
    alpha1 = 0.05,
    verbose=False,
    getMag = True
)

t2 = time.perf_counter()

print(f"LiNGAM-SF took {t1 - t0:.4f} seconds")
print("g0, OK!")




    
    


    
    