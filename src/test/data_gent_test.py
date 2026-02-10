# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 12:42:24 2026

@author: chdem
"""

from __future__ import annotations


import csv
import time
import gc
import numpy as np
import pandas as pd

from causallearn.graph.GeneralGraph import GeneralGraph


# If you already have dag2pag in your repo, use it.
# Example: from src.dag2pag import dag2pag
from causaldiscovery.graphs.dag2pag import dag2pag
from causaldiscovery.algorithms.FCI_SF import fci_sf
from causaldiscovery.CItest.noCache_CI_Test import myTest


# -----------------------------
# Configuration
# -----------------------------
NUM_VARS = 50
EXPECTED_DEGREE = 2          # Expected degree of the Erdos-Renyi graph
NUM_RANDOM_DAGS = 100
NUM_REPEATS_PER_DAG = 3      # Repeat with different seeds for stability
ALPHA = 0.05

START_N = 375
MAX_N = 24000

OUT_CSV = "fci_convergence_results.csv"


# -----------------------------
# Utilities: DAG generation + linear Gaussian SEM simulation
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


def sample_linear_gaussian_sem(A: np.ndarray, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """
    Linear SEM: X = B^T X + e, where B follows A (edge weights) and e is Gaussian noise.
    The data are generated following the topological order induced by the DAG A.
    """
    n_vars = A.shape[0]
    # Edge weights: avoid values too close to zero
    B = np.zeros_like(A, dtype=float)
    for i in range(n_vars):
        for j in range(n_vars):
            if A[i, j] == 1:
                w = rng.uniform(0.5, 2.0) * (1 if rng.random() < 0.5 else -1)
                B[i, j] = w

    # Topological order
    topo = topological_order(A)

    X = np.zeros((n_samples, n_vars), dtype=float)
    noise = rng.normal(loc=0.0, scale=1.0, size=(n_samples, n_vars))

    for v in topo:
        parents = np.where(A[:, v] == 1)[0]
        if len(parents) == 0:
            X[:, v] = noise[:, v]
        else:
            X[:, v] = X[:, parents] @ B[parents, v] + noise[:, v]

    # Standardize variables (Fisher-Z usually works better on scaled data)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
    return X


def topological_order(A: np.ndarray) -> list[int]:
    n = A.shape[0]
    indeg = A.sum(axis=0).astype(int).tolist()
    q = [i for i in range(n) if indeg[i] == 0]
    out = []
    while q:
        u = q.pop()
        out.append(u)
        for v in np.where(A[u, :] == 1)[0]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(int(v))
    if len(out) != n:
        raise ValueError("The graph is not a DAG (cycle detected).")
    return out


# -----------------------------
# Utilities: extract adjacencies / orientations from GeneralGraph (PAG)
# -----------------------------
def graph_to_skeleton_adj(G: GeneralGraph) -> np.ndarray:
    """
    Returns the undirected adjacency matrix (0/1) of the skeleton
    of a causallearn GeneralGraph.
    """
    nodes = G.get_nodes()
    p = len(nodes)
    adj = np.zeros((p, p), dtype=int)
    for i in range(p):
        for j in range(i + 1, p):
            e = G.get_edge(nodes[i], nodes[j])
            if e is not None:
                adj[i, j] = 1
                adj[j, i] = 1
    return adj


def graph_to_arrowheads(G: GeneralGraph) -> np.ndarray:
    """
    arrow[i, j] = 1 if there is an arrowhead pointing to j
    on the edge i -?-> j.
    In causallearn PAGs, the endpoint at the destination node
    can be of type ARROW.
    This does NOT fully capture all PAG nuances (e.g., circles),
    but it serves as a reasonable proxy.
    """
    from causallearn.graph.Endpoint import Endpoint

    nodes = G.get_nodes()
    p = len(nodes)
    arrow = np.zeros((p, p), dtype=int)
    for i in range(p):
        for j in range(p):
            if i == j:
                continue
            e = G.get_edge(nodes[i], nodes[j])
            if e is None:
                continue
            # We query the endpoint at node j for the edge (i, j)
            end_j = G.get_endpoint(nodes[i], nodes[j])
            if end_j == Endpoint.ARROW:
                arrow[i, j] = 1
    return arrow


# -----------------------------
# Metrics
# -----------------------------
def precision_recall(tp: int, fp: int, fn: int) -> tuple[float, float]:
    prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    return prec, rec


def skeleton_metrics(adj_true: np.ndarray, adj_est: np.ndarray) -> dict:
    # Work only with the upper triangular part to avoid double counting
    triu = np.triu_indices_from(adj_true, k=1)
    t = adj_true[triu]
    e = adj_est[triu]

    tp = int(np.sum((t == 1) & (e == 1)))
    fp = int(np.sum((t == 0) & (e == 1)))
    fn = int(np.sum((t == 1) & (e == 0)))

    prec, rec = precision_recall(tp, fp, fn)

    # SHD (skeleton only): number of edge additions and deletions
    shd = fp + fn

    return {
        "skel_tp": tp,
        "skel_fp": fp,
        "skel_fn": fn,
        "skel_precision": prec,
        "skel_recall": rec,
        "skel_shd": shd,
    }


def arrow_metrics(arr_true: np.ndarray, arr_est: np.ndarray,
                  adj_true: np.ndarray, adj_est: np.ndarray) -> dict:
    """
    Simple score for arrowheads:
    - Only evaluate pairs that are connected either in the true graph
      or in the estimated graph (their union),
      to avoid trivially rewarding empty graphs.
    """
    p = arr_true.shape[0]
    mask = ((adj_true + adj_est) > 0).astype(bool)
    np.fill_diagonal(mask, False)

    t = arr_true[mask]
    e = arr_est[mask]

    tp = int(np.sum((t == 1) & (e == 1)))
    fp = int(np.sum((t == 0) & (e == 1)))
    fn = int(np.sum((t == 1) & (e == 0)))

    prec, rec = precision_recall(tp, fp, fn)

    return {
        "arr_tp": tp,
        "arr_fp": fp,
        "arr_fn": fn,
        "arr_precision": prec,
        "arr_recall": rec,
    }


# -----------------------------
# Main loop
# -----------------------------
def dataset_sizes(start_n: int, max_n: int) -> list[int]:
    out = []
    n = start_n
    while n <= max_n:
        out.append(n)
        n *= 2
    return out


def run():
    sizes = dataset_sizes(START_N, MAX_N)
    print("Dataset sizes:", sizes)

    header = [
        "dag_id", "repeat_id", "n", "alpha",
        "fci_time_sec",
        "skel_precision", "skel_recall", "skel_shd",
        "arr_precision", "arr_recall",
    ]

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)

    base_rng = np.random.default_rng(123)

    for dag_id in range(NUM_RANDOM_DAGS):
        # 1) Fix a DAG
        rng_dag = np.random.default_rng(base_rng.integers(0, 2**32 - 1))
        A = random_dag_erdos(NUM_VARS, EXPECTED_DEGREE, rng_dag)

        # Variable names
        names = [f"X{i}" for i in range(NUM_VARS)]

        # Ground-truth PAG from the true DAG
        # If dag2pag expects a different input format,
        # adapt here (e.g., convert A to a networkx.DiGraph with names).
        import networkx as nx
        digraph = nx.DiGraph()
        digraph.add_nodes_from(names)
        for i in range(NUM_VARS):
            for j in range(NUM_VARS):
                if A[i, j] == 1:
                    digraph.add_edge(names[i], names[j])

        G_true = dag2pag(digraph, names)  # GeneralGraph
        adj_true = graph_to_skeleton_adj(G_true)
        arr_true = graph_to_arrowheads(G_true)

        for rep in range(NUM_REPEATS_PER_DAG):
            rng_rep = np.random.default_rng(base_rng.integers(0, 2**32 - 1))

            for n in sizes:
                # 2) Sample data from the SAME DAG with n instances
                X = sample_linear_gaussian_sem(A, n, rng_rep)
                df = pd.DataFrame(X, columns=names)

                # 3) Run FCI
                t0 = time.time()
                # fci() accepts a numpy matrix + CI test; fisherz operates on the matrix
                # Note: causallearn.fci uses its own wrapper; if you use myTest(df),
                # you can replace the independence_test_method here.
                """
                G_est, _ = fci(
                    X,
                    independence_test_method=fisherz,
                    alpha=ALPHA,
                    verbose=False
                )
                """
                CI_test = myTest(df)
                output_fci_stable = fci_sf(
                    X,
                    independence_test_method=CI_test,
                    initial_sep_sets={},
                    alpha=ALPHA,
                    initial_graph=GeneralGraph([]),
                    new_node_names=names,
                    verbose=False
                )
                G_est = output_fci_stable[0]
                t1 = time.time()

                # 4) Metrics
                adj_est = graph_to_skeleton_adj(G_est)
                arr_est = graph_to_arrowheads(G_est)

                m_skel = skeleton_metrics(adj_true, adj_est)
                m_arr = arrow_metrics(arr_true, arr_est, adj_true, adj_est)

                row = [
                    dag_id, rep, n, ALPHA,
                    (t1 - t0),
                    m_skel["skel_precision"], m_skel["skel_recall"], m_skel["skel_shd"],
                    m_arr["arr_precision"], m_arr["arr_recall"],
                ]

                with open(OUT_CSV, "a", newline="") as f:
                    csv.writer(f).writerow(row)

                print(
                    f"DAG {dag_id} rep {rep} n={n}  "
                    f"time={t1-t0:.2f}s  "
                    f"skelP/R={m_skel['skel_precision']:.2f}/{m_skel['skel_recall']:.2f}  "
                    f"arrP/R={m_arr['arr_precision']:.2f}/{m_arr['arr_recall']:.2f}"
                )

                del X, df, G_est, adj_est, arr_est
                gc.collect()

        del A, G_true, adj_true, arr_true, digraph
        gc.collect()

    print(f"Done. Results in: {OUT_CSV}")


if __name__ == "__main__":
    run()
