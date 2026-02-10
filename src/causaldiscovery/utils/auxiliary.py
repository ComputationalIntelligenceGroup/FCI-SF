from causallearn.graph.GeneralGraph import GeneralGraph
from typing import List, Tuple
from itertools import combinations

import numpy as np
import random
import pandas as pd
import networkx as nx


# pgmpy: Linear Gaussian Bayesian Network
from pgmpy.models import LinearGaussianBayesianNetwork






def make_bn_truth_and_sample(
    *,
    base_rng: np.random.Generator,
    num_vars: int,
    expected_degree: float,
    n_samples: int
) -> Tuple[ nx.DiGraph, pd.DataFrame]:
    """
    Generates:
      - digraph: networkx.DiGraph for ground truth
      - df: sampled data from the pgmpy model using the provided seed_data


    Returns:
      A, model, digraph, df, names, G_true, adj_true, arr_true
    """
    # 1) Generate random DAG adjacency using the exact RNG pattern you specified
    rng_dag = np.random.default_rng(int(base_rng.integers(0, 2**32 - 1)))
    A = random_dag_erdos(num_vars, expected_degree, rng_dag)

    names = [f"X{i}" for i in range(num_vars)]

    # 2) Build networkx DiGraph (ground truth)
    digraph = nx.DiGraph()
    digraph.add_nodes_from(names)
    for i in range(num_vars):
        for j in range(num_vars):
            if A[i, j] == 1:
                digraph.add_edge(names[i], names[j])

    # 3) Build pgmpy linear Gaussian BN with random CPDs (seeded from base_rng)
    seed_model = int(base_rng.integers(0, 2**32 - 1))
    model = pgmpy_lgbn_from_adj(A, names, seed=seed_model)

      
    
    # 4) Sample data using the seed_data you pass in
    seed_data = int(base_rng.integers(0, 2**32 - 1))
    df = simulate_from_pgmpy(model, n_samples, seed_data, names)

    return digraph, df


# -----------------------------
# Utilities: DAG generation
# -----------------------------
def random_dag_erdos(n: int, expected_degree: float, rng: np.random.Generator) -> np.ndarray:
    """
    Returns an adjacency matrix (n x n) of a random Erdos-Renyi DAG
    using a random topological ordering to guarantee acyclicity.
    """
    # Approximate probability for the expected degree in a DAG (only i < j)
    p = min(1.0, expected_degree / max(1, (n - 1)))
    order = rng.permutation(n)
    A = np.zeros((n, n), dtype=int)

    # Only allow forward edges according to the random order
    for ii in range(n):
        for jj in range(ii + 1, n):
            if rng.random() < p:
                u = order[ii]
                v = order[jj]
                A[u, v] = 1
    return A


def pgmpy_lgbn_from_adj(A: np.ndarray, names: list[str], seed: int) -> LinearGaussianBayesianNetwork:
    """
    Build a pgmpy LinearGaussianBayesianNetwork from an adjacency matrix A (parents -> child),
    and generate random Linear Gaussian CPDs for it.
    """
    edges = []
    p = A.shape[0]
    for i in range(p):
        for j in range(p):
            if A[i, j] == 1:
                edges.append((names[i], names[j]))

    model = LinearGaussianBayesianNetwork(edges)
    model.add_nodes_from(names)  # <- CLAVE: mete también los nodos aislados
    model.get_random_cpds(loc=0.0, scale=1.0, inplace=True, seed=seed)
    model.check_model()
    return model


def simulate_from_pgmpy(model: LinearGaussianBayesianNetwork, n_samples: int, seed: int, names: list[str]) -> pd.DataFrame:
    """
    Sample n_samples iid rows from the pgmpy linear Gaussian BN.
    Then z-score columns (optional but keeps parity with your previous pipeline).
    """
    df = model.simulate(n_samples, seed=seed)
    df = df.reindex(columns=names)
    df = df.astype(float)
    df = (df - df.mean(axis=0)) / (df.std(axis=0) + 1e-12)
    return df



def get_numerical_edges(G: GeneralGraph) -> List[Tuple[int, int]]:
    """Returns all the edges from the general graph"""
    res: List[Tuple[int, int]] = []
    for edge in G.get_graph_edges():
        numerical_edge = (G.node_map[edge.get_node1()], G.node_map[edge.get_node2()])
        res.append(numerical_edge)
    
    return res




def apply_permutation(a, p_rows=None, p_cols=None):
    """
    a: ndarray
    p_rows: integer array such that p_rows[i] = destiny of row i
    p_cols:  integer array such that p_cols[j] =  destiny of column j
    """
    out = a
    if p_rows is not None:
        p_rows = np.asarray(p_rows)
        # Basic check
        assert sorted(p_rows) == list(range(a.shape[0])), "p_rows must be a permutation of 0..nrows-1"
        order_rows = np.argsort(p_rows)   # inversa: en la nueva pos j, quién venía
        out = out[order_rows, ...]
    if p_cols is not None:
        p_cols = np.asarray(p_cols)
        assert sorted(p_cols) == list(range(a.shape[1])), "p_cols  must be a permutation of 0..ncols-1"
        order_cols = np.argsort(p_cols)
        out = out[..., order_cols]
    return out



def random_permutation(size: int) -> List[int]:
    
    numbers = [i for i in range(size)]
    res = []
    
    for _ in range(size):
        pos = random.random.randint(0, len(numbers)-1)
        elem = numbers.pop(pos)
        res.append(elem)
        
    return res


def powerset(s):
    """All subsets of a set s (as sets)."""
    s = list(s)
    for r in range(len(s) + 1):
        for comb in combinations(s, r):
            yield set(comb)
    
    

        