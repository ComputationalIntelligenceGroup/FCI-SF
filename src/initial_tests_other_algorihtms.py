#!/usr/bin/env python3

from causallearn.utils.cit           import fisherz      # ∼O(n³) linear-Gaussian CI test
from causallearn.utils.GraphUtils    import GraphUtils
import dowhy.datasets as ds          # DoWhy ≥ 0.10

import numpy as np
from skfeature.utility.mutual_information import su_calculation

from CSBS_FS import csbs_fs
from PRCDSF_FS import prcdsf_fs
from S_CDFSF_FS import s_cdfsf_fs
from CSSU_FS import cssu_fs
import hill_climbing as hc
from FCI_FS import fci_fs

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


gen = AcyclicGraphGenerator('linear', npoints=50, nodes=100, dag_type='erdos', expected_degree= 2)
df, G = gen.generate()   # X: DataFrame, G: networkx.DiGraph

draw_digraph(G, "Ground-Truth DAG")

data   = df.values                   # numpy array, n×p
names = df.columns


data1 = data[:, 0:50]
names1 = names[ 0:50]
names2 = names[ 50: ]

"""
from pgmpy.utils import get_example_model
from pgmpy.sampling import BayesianModelSampling

# Load the Survey network (works if your pgmpy version includes it)
model = get_example_model("survey")

# Initialize a sampler
sampler = BayesianModelSampling(model)

# Generate 100 random samples
data = sampler.forward_sample(size=5000).to_numpy()


data1 = data[:, 0:4]
names = ["A", "S", "B", "E", "O", "R", "T"]
names1 = names[0:4]
names2 = names[4:6]
"""

k = 4

g0, nCI0, avg_ss0, exec_t0 = csbs_fs(data1,independence_test_method=fisherz, new_node_names= names1 ,verbose = True,  
            max_iter = 1e3) 
print("g0, OK!")
g1, _, _, exec_t1 = prcdsf_fs(data1,independence_test_method= fisherz,  new_node_names= names1 ,verbose = True,  
            max_iter = 1e3)
print("g1, OK!")
g2, _, _, exec_t2 = s_cdfsf_fs(data1,independence_test_method=fisherz , new_node_names= names1 ,verbose = True,  
            max_iter = 1e3)
print("g2, OK!")
#g3, _, _, exec_t3 = cssu_fs(data1, alpha= 5e-2,  new_node_names= names1 ,verbose = False,  max_iter = 1e3)
#print("g3, OK!")
g4, _, _, _, _, exec_t4 = fci_fs(data1, independence_test_method=fisherz ,  new_node_names= names1 ,verbose = False)
print("g4, OK!")

graphs = [g0, g1, g2, g4]


for i in range(k):
    GraphUtils.to_pydot(graphs[i]).write_png(f"testing_marginal_OAlg-1-{i}.png")
    
    
g0, _, _, exec_t02 = csbs_fs(data,independence_test_method=fisherz, initial_graph= g0, new_node_names= names2 ,verbose = True,  
            max_iter = 1e3) 
print("g0, OK!")
g1, _, _, exec_t12 = prcdsf_fs(data,independence_test_method=fisherz, initial_graph= g1 ,  new_node_names= names2 ,verbose = True,  
            max_iter = 1e3)
print("g1, OK!")
g2, _, _, exec_t22 = s_cdfsf_fs(data,independence_test_method=fisherz, initial_graph= g2 ,  new_node_names= names2 ,verbose = True,  
            max_iter = 1e3)
print("g2, OK!")
#g3, _, _, exec_t32 = cssu_fs(data, alpha=  1e-4,  new_node_names= names2, initial_graph= g3 ,verbose = False,  max_iter = 1e3)
#print("g3, OK!")
g4, _, _, _, _, exec_t42 = fci_fs(data, independence_test_method=fisherz, initial_graph= g4 ,  new_node_names= names2 ,verbose = False)
print("g4, OK!")



for i in range(k):
    GraphUtils.to_pydot(graphs[i]).write_png(f"testing_full_OAlg-1-{i}.png")
    
    