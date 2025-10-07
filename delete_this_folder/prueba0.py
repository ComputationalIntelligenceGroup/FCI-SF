#!/usr/bin/env python3

from causallearn.utils.cit           import fisherz      # ∼O(n³) linear-Gaussian CI test
from causallearn.utils.GraphUtils    import GraphUtils
import dowhy.datasets as ds          # DoWhy ≥ 0.10

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

data_dict = ds.dataset_from_random_graph(
    num_vars=8,          # number of variables
    num_samples=50,    # rows
    prob_edge=0.25,      # expected edge density
    random_seed=42
)
df         = data_dict["df"]         # pandas DataFrame
true_gml   = data_dict["gml_graph"]  # ground-truth DAG in GML format


data   = df.values                   # numpy array, n×p
pag, edges    = fci(data, fisherz, alpha=0.05)   # returns a PAG (partial ancestral graph)



# Optional: visualise (requires graphviz & pydot)
GraphUtils.to_pydot(pag, labels=df.columns).write_png("learned_pag2.png")

