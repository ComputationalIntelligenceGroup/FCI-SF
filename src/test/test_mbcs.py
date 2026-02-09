#!/usr/bin/env python3

from causallearn.utils.cit           import fisherz      # ∼O(n³) linear-Gaussian CI test
from causallearn.utils.GraphUtils    import GraphUtils
import dowhy.datasets as ds          # DoWhy ≥ 0.10


import numpy as np
import time



from causaldiscovery.algorithms.MBCS import mbcs_skeleton
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


gen = AcyclicGraphGenerator('linear', npoints=100, nodes=200, dag_type='erdos', expected_degree= 2)
df, G = gen.generate()   # X: DataFrame, G: networkx.DiGraph

draw_digraph(G, "Ground-Truth DAG")

data   = df.values                   # numpy array, n×p
names = df.columns

CI_test = myTest(df)


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

k = 1


t0 = time.perf_counter()
g0 = mbcs_skeleton(
    data,
    independence_test_method=CI_test,
    new_node_names=names,
    verbose=False
)
t1 = time.perf_counter()

print(f"mbcs_skeleton took {t1 - t0:.4f} seconds")
print("g0, OK!")

graphs = [g0]


for i in range(k):
    GraphUtils.to_pydot(graphs[i].G).write_png(f"testing_marginal_OAlg-1-{i}.png")
    
    


    
    