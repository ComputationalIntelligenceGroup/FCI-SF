# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 13:51:32 2026

@author: chdem
"""

from __future__ import annotations

import csv
import time
import gc
import numpy as np
import pandas as pd

from causallearn.graph.GeneralGraph import GeneralGraph

from causaldiscovery.graphs.dag2pag import dag2pag
from causaldiscovery.algorithms.FCI_SF import fci_sf
from causaldiscovery.algorithms.CSSU import cssu
from causaldiscovery.CItest.noCache_CI_Test import myTest

# pgmpy: Linear Gaussian Bayesian Network
from pgmpy.models import LinearGaussianBayesianNetwork


# -----------------------------
# Configuration
# -----------------------------
NUM_VARS = 50
EXPECTED_DEGREE = 2          # Expected degree of the Erdos-Renyi graph
NUM_RANDOM_DAGS = 20
NUM_REPEATS_PER_DAG = 3      # Repeat with different seeds for stability
ALPHA = 0.05
MAX_ITER = int(1e3)

START_N = 375
MAX_N = 24000

OUT_CSV = "fci_convergence_results2.csv"

#base random generator to guarantee reproducibility

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


def dataset_sizes(start_n: int, max_n: int) -> list[int]:
    out = []
    n = start_n
    while n <= max_n:
        out.append(n)
        n *= 2
    return out


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
    triu = np.triu_indices_from(adj_true, k=1)
    t = adj_true[triu]
    e = adj_est[triu]

    tp = int(np.sum((t == 1) & (e == 1)))
    fp = int(np.sum((t == 0) & (e == 1)))
    fn = int(np.sum((t == 1) & (e == 0)))

    prec, rec = precision_recall(tp, fp, fn)
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
def run():
    sizes = dataset_sizes(START_N, MAX_N)
    print("Dataset sizes:", sizes)

    header = [
        "dag_id", "repeat_id", "n", "alpha",

        "fci_time_sec",
        "fci_skel_precision", "fci_skel_recall", "fci_skel_shd",
        "fci_arr_precision", "fci_arr_recall",

        "cssu_time_sec",
        "cssu_skel_precision", "cssu_skel_recall", "cssu_skel_shd",
        "cssu_arr_precision", "cssu_arr_recall",
    ]

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)

    base_rng = np.random.default_rng(123)

    for dag_id in range(NUM_RANDOM_DAGS):
        # 1) Fix a random DAG structure (A) once
        rng_dag = np.random.default_rng(base_rng.integers(0, 2**32 - 1))
        A = random_dag_erdos(NUM_VARS, EXPECTED_DEGREE, rng_dag)

        names = [f"X{i}" for i in range(NUM_VARS)]

        # Build networkx DiGraph for dag2pag ground truth
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
            # 2) Fix a pgmpy Linear Gaussian BN with random CPDs (fixed across n)
            seed_model = int(base_rng.integers(0, 2**32 - 1))
            model = pgmpy_lgbn_from_adj(A, names, seed=seed_model)

            for n in sizes:
                # 3) Sample data from the SAME BN with n instances
                seed_data = int(base_rng.integers(0, 2**32 - 1))
                df = simulate_from_pgmpy(model, n_samples=n, seed=seed_data, names=names)

                # --- Run FCI_SF (stable) ---
                t0 = time.time()
                CI_test = myTest(df)
                output_fci_stable = fci_sf(
                    df.to_numpy(copy=False),
                    independence_test_method=CI_test,
                    initial_sep_sets={},
                    alpha=ALPHA,
                    initial_graph=GeneralGraph([]),
                    new_node_names=names,
                    verbose=False
                )
                G_fci = output_fci_stable[0]
                t1 = time.time()
                fci_time = t1 - t0

                adj_fci = graph_to_skeleton_adj(G_fci)
                arr_fci = graph_to_arrowheads(G_fci)
                m_fci_skel = skeleton_metrics(adj_true, adj_fci)
                m_fci_arr = arrow_metrics(arr_true, arr_fci, adj_true, adj_fci)

                # --- Run CSSU ---
                t2 = time.time()
                output_cssu = cssu(
                    df.to_numpy(copy=False),
                    alpha=ALPHA,
                    initial_graph=GeneralGraph([]),
                    new_node_names=names,
                    verbose=False,
                    max_iter=MAX_ITER
                )
                G_cssu = output_cssu[0] if isinstance(output_cssu, (tuple, list)) else output_cssu
                t3 = time.time()
                cssu_time = t3 - t2

                adj_cssu = graph_to_skeleton_adj(G_cssu)
                arr_cssu = graph_to_arrowheads(G_cssu)
                m_cssu_skel = skeleton_metrics(adj_true, adj_cssu)
                m_cssu_arr = arrow_metrics(arr_true, arr_cssu, adj_true, adj_cssu)

                row = [
                    dag_id, rep, n, ALPHA,

                    fci_time,
                    m_fci_skel["skel_precision"], m_fci_skel["skel_recall"], m_fci_skel["skel_shd"],
                    m_fci_arr["arr_precision"], m_fci_arr["arr_recall"],

                    cssu_time,
                    m_cssu_skel["skel_precision"], m_cssu_skel["skel_recall"], m_cssu_skel["skel_shd"],
                    m_cssu_arr["arr_precision"], m_cssu_arr["arr_recall"],
                ]

                with open(OUT_CSV, "a", newline="") as f:
                    csv.writer(f).writerow(row)

                print(
                    f"DAG {dag_id} rep {rep} n={n} | "
                    f"FCI time={fci_time:.2f}s skelP/R={m_fci_skel['skel_precision']:.2f}/{m_fci_skel['skel_recall']:.2f} SHD={m_fci_skel['skel_shd']} | "
                    f"CSSU time={cssu_time:.2f}s skelP/R={m_cssu_skel['skel_precision']:.2f}/{m_cssu_skel['skel_recall']:.2f} SHD={m_cssu_skel['skel_shd']}"
                )

                del df, G_fci, G_cssu, adj_fci, adj_cssu, arr_fci, arr_cssu
                gc.collect()

            del model
            gc.collect()

        del A, G_true, adj_true, arr_true, digraph
        gc.collect()

    print(f"Done. Results in: {OUT_CSV}")


if __name__ == "__main__":
    run()
