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

def load_module_from(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod   
    spec.loader.exec_module(mod)
    return mod

fci_path = r"C:\Users\chdem\0UNIVERSIDAD\CIG\code\causal-learn\causallearn\search\ConstraintBased\FCI.py"


fci_mod = load_module_from(fci_path, "fci_mod")

from fci_mod import fci

import networkx as nx
import matplotlib.pyplot as plt
from io import StringIO


data_dict = ds.dataset_from_random_graph(
    num_vars=26,          # number of variables
    num_samples=1000,    # rows
    prob_edge=1,      # expected edge density
    random_seed= 68,
    prob_type_of_data= [1, 0, 0]
    
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

k = 5

g0, nCI0, avg_ss0, exec_t0 = csbs_fs(data1,independence_test_method="chisq", new_node_names= names1 ,verbose = False) 
g1, _, _, _ = prcdsf_fs(data1,independence_test_method="chisq",  new_node_names= names1 ,verbose = False)
g2, _, _, _ = s_cdfsf_fs(data1,independence_test_method="chisq",  new_node_names= names1 ,verbose = False)
g3, _, _, _ = cssu_fs(data1, alpha= 5e-2,  new_node_names= names1 ,verbose = False)
g4, _, _, _, _, _ = fci_fs(data1, independence_test_method="chisq" ,  new_node_names= names1 ,verbose = False)

graphs = [g0, g1, g2, g3, g4]


for i in range(k):
    GraphUtils.to_pydot(graphs[i]).write_png(f"testing_marginal_OAlg-1-{i}.png")
    
    
g0, _, _, _ = csbs_fs(data,independence_test_method="chisq", initial_graph= g0, new_node_names= names2 ,verbose = False) 
g1, _, _, _ = prcdsf_fs(data,independence_test_method="chisq", initial_graph= g1 ,  new_node_names= names2 ,verbose = False)
g2, _, _, _ = s_cdfsf_fs(data,independence_test_method="chisq", initial_graph= g2 ,  new_node_names= names2 ,verbose = False)
g3, _, _, _ = cssu_fs(data, alpha=  1e-4,  new_node_names= names2, initial_graph= g3 ,verbose = False)
g4, _, _, _, _, _ = fci_fs(data, independence_test_method="chisq", initial_graph= g4 ,  new_node_names= names2 ,verbose = False)




for i in range(k):
    GraphUtils.to_pydot(graphs[i]).write_png(f"testing_full_OAlg-1-{i}.png")
    
    