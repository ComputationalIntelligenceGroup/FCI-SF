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




import importlib.util, sys

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from cdt.data import AcyclicGraphGenerator


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


gen = AcyclicGraphGenerator('linear', npoints=12000, nodes=50, dag_type='erdos', expected_degree= 1)
df, G = gen.generate()   # X: DataFrame, G: networkx.DiGraph

draw_digraph(G, "Ground-Truth DAG")

data   = df.values                   # numpy array, n×p
names = df.columns

CI_test = myTest(df)

t0 = time.perf_counter()
g0, Lt,T, numCItest = lingam_sf(
    data,
    independence_test_method=CI_test,
    new_node_names=names,
    alpha1 = 0.05,
    verbose=False
)
t1 = time.perf_counter()

print(f"LiNGAM-SF took {t1 - t0:.4f} seconds")
print("g0, OK!")




    
    


    
    