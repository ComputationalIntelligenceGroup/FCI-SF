#!/usr/bin/env python3

from causallearn.utils.cit import *
from causallearn.utils.GraphUtils    import GraphUtils
import dowhy.datasets as ds          # DoWhy ≥ 0.10

from typing import List

from FCI_FS import fci_fs

import networkx as nx
import matplotlib.pyplot as plt
from io import StringIO


data_dict = ds.dataset_from_random_graph(
    num_vars=26,          # number of variables
    num_samples=100,    # rows
    prob_edge=0.7,      # expected edge density
    random_seed= 69
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

names = df.columns

data1 = data[:, 0:13]
names1 = names[ 0:13]
names2 = names[ 13: ]








graph1, edgePropList, sep_sets = fci_fs(data1, "fisherz", depth = 3, max_path_length = 5, new_node_names=names1)

GraphUtils.to_pydot(graph1).write_png("FCI-test-1.png")


graph2, edgePropList, sep_sets = fci_fs(data, "fisherz", depth = 3, max_path_length = 5, initial_graph= graph1, initial_sep_sets=sep_sets, new_node_names=names2, verbose = False)

GraphUtils.to_pydot(graph1).write_png("FCI-test-2.png")





