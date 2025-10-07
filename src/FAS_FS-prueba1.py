#!/usr/bin/env python3

from causallearn.utils.cit import *
from causallearn.utils.GraphUtils    import GraphUtils
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.Node import Node
import dowhy.datasets as ds          # DoWhy ≥ 0.10

from typing import List

from FAS_FS import fas_fs

import networkx as nx
import matplotlib.pyplot as plt
from io import StringIO


data_dict = ds.dataset_from_random_graph(
    num_vars=8,          # number of variables
    num_samples=100,    # rows
    prob_edge=0.5      # expected edge density
)
df         = data_dict["df"]         # pandas DataFrame
true_gml   = data_dict["gml_graph"]  # ground-truth DAG in GML format

# Load graph from GML string
G = nx.parse_gml(StringIO(true_gml))
plt.figure(figsize=(6, 6))
nx.draw_networkx(G, with_labels=True, node_size=800, node_color="lightblue", arrows=True)
plt.title("Ground-Truth DAG")
plt.show()

data   = df.values                   # numpy array, n×p

data1 = data[:, 0:4]

data2 = data[:, 4:]

nodes: List[Node] = []



for name in df.columns:
        nodes.append(GraphNode(name))

nodes1 = nodes[0:4]
nodes2 = nodes[4:]

independence_test_method = CIT(data, method="fisherz")

pag, sepsets, test_results    = fas_fs(data = data1,independence_test_method = independence_test_method, alpha=0.05, depth = 3, stable = True, verbose=True)   # returns a PAG (partial ancestral graph)

GraphUtils.to_pydot(pag).write_png("incremental_pag1-0.png")

pag, sepsets, test_results    = fas_fs(data, independence_test_method, initial_sep_sets=sepsets, initial_graph= pag, alpha=0.05, depth = 3, stable = True, verbose=True)   # returns a PAG (partial ancestral graph)

GraphUtils.to_pydot(pag).write_png("incremental_pag2-0.png")




