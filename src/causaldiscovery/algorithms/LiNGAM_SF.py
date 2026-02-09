# -*- coding: utf-8 -*-
"""
Created on Sun Feb  8 16:19:49 2026

@author: chdem
"""

from __future__ import annotations

import warnings
from typing import Dict, FrozenSet, Tuple, List
from numpy import ndarray
import numpy as np
import time
import pandas as pd

from causallearn.graph.Graph import Graph
from causallearn.graph.GeneralGraph import GeneralGraph 

from causallearn.utils.cit import *

import sys, pathlib
repo_root = pathlib.Path(__file__).resolve().parents[3]
sys.path.append(str(repo_root / "src"))

from causaldiscovery.CItest.noCache_CI_Test import myTest
from causaldiscovery.graphs.IncrementalGraph import IncrementalGraph
from causaldiscovery.utils.orientation import orientation
from causaldiscovery.algorithms.MBCS import mbcs_skeleton
from causaldiscovery.algorithms.CSBS import csbs
from causaldiscovery.CItest.noCache_CI_Test import myTest
import networkx as nx


# mb.G, num_CI_tests, avg_sepset_size, total_exec_time


def lingam_sf(data: ndarray, independence_test_method: CIT_Base, alpha1: float = 0.05,  alpha2: float = 0.001,  r = 0.7,
            initial_graph: GeneralGraph = None,  verbose: bool = False,  new_node_names:List[str] = None, **kwargs) -> Tuple[Graph, int, int]:
    
    nCI = 0 
    
    if initial_graph is None:
        initial_graph = GeneralGraph([])
        
    Gt1 = IncrementalGraph(0, initial_graph)
    
    if Gt1.G.get_num_nodes() == 0:
        Gt1, num_CI_tests, sepset_size = mbcs_skeleton(data = data, independence_test_method = independence_test_method, alpha = alpha1, 
                           verbose = verbose, new_node_names = new_node_names)
        nCI += num_CI_tests
    else:
        Gt1, num_CI_tests, sepset_size = csbs(data, independence_test_method=independence_test_method, alpha= alpha1, 
                           initial_graph= initial_graph, new_node_names= new_node_names ,verbose = verbose, 
                           max_iter = 0, only_skel = True) 
        nCI += num_CI_tests
        
    
    print("Skeleton done!")
         
    Gt2, Lt2,T, num_CI_tests = icdplv(data,Gt1,  r, alpha2, verbose = verbose)
    
    nCI += num_CI_tests
        
    return Gt2, Lt2, T, num_CI_tests

def compute_residual(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Residual r_x(y) = x - (cov(x,y)/var(y)) * y
    x, y: shape (n_samples,)
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    # center
    xc = x - x.mean()
    yc = y - y.mean()

    var_y = np.mean(yc * yc) + eps
    cov_xy = np.mean(xc * yc)
    beta = cov_xy / var_y

    return x - beta * y

def judge_direction(row1: np.ndarray, row2: np.ndarray) -> int:
    """
    row1, row2: 1D binary arrays (support rows) with same length
    Returns:
      +1 means var1 -> var2
      -1 means var2 -> var1
       0 undecided
    """
    row1 = np.asarray(row1).astype(int).reshape(-1)
    row2 = np.asarray(row2).astype(int).reshape(-1)
    assert row1.shape == row2.shape

    n0_star = int(np.sum((row1 == 0) & (row2 == 1)))
    nstar_0 = int(np.sum((row1 == 1) & (row2 == 0)))

    if (n0_star > 0) and (nstar_0 == 0):
        return +1
    if (nstar_0 > 0) and (n0_star == 0):
        return -1
    return 0




def icdplv(
    data: ndarray,
    incremental_graph,
    r = 0.7,
    sig = 0.001,
    verbose = False,
    orientation_kwargs=None,
):
    """
    Parameters
    ----------
    incremental_graph : your object
        Must provide:
    X : ndarray (n_samples, p)
    r : float sampling ratio passed to orientation()
    sig : float significance threshold for Proposition 1
    orientation_kwargs : dict forwarded to orientation()

    Returns
    -------
    Gt : nx.DiGraph
        Directed edges accepted by Prop1 and Prop2
    Lalter : set
        Endpoints of unresolved edges after Prop1 (candidate latent confounders)
    num_CI_tests: int
        Number of performed CI test
    """
    if orientation_kwargs is None:
        orientation_kwargs = {}
        
    X = np.asarray(data, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError("data must be 2D array (n_samples, p)")
    n_samples, p = X.shape
    
 

    if not (0 < r <= 1):
        raise ValueError("r must be in (0,1]")
        
    num_CI_tests = 0

    node_names = incremental_graph.G.get_node_names()
    Gt = nx.DiGraph()
    Gt.add_nodes_from(node_names)

    T = []          # unresolved edges after Prop1
    Lalter = set()  # endpoints of unresolved edges after Prop1

    # -------------------------
    # Phase A: Proposition 1 
    # -------------------------
    U = incremental_graph.get_numerical_edges()

    for (a, b) in U:
        xa = X[:, a]
        xb = X[:, b]
        
        

        ra_b = compute_residual(xa, xb)  
        rb_a = compute_residual(xb, xa)  

        # one DF per edge, then one CI tester
        df_edge = pd.DataFrame({
            "xa": xa,
            "xb": xb,
            "ra_b": ra_b,
            "rb_a": rb_a,
        })
        ci = myTest(df_edge)

        idx_xa, idx_xb, idx_ra_b, idx_rb_a = 0, 1, 2, 3

        p1 = ci(idx_ra_b, idx_xb, set())  # ra_b ⟂ xb ?
        p2 = ci(idx_rb_a, idx_xa, set())  # rb_a ⟂ xa ?
        
        num_CI_tests += 2

        na = node_names[a]
        nb = node_names[b]

        if (p1 > sig) and (p2 <= sig):
            print("b -> a")
            # b -> a
            Gt.add_edge(nb, na)
        elif (p2 > sig) and (p1 <= sig):
            print("a -> b")
            # a -> b
            Gt.add_edge(na, nb)
        else:
            print("Common confounder (append T)")
            T.append((a, b))
            Lalter.add(a)
            Lalter.add(b)

    # -------------------------
    # Phase B: Proposition 2 
    # -------------------------
    def add_edge_if_no_cycle(u, v):
        # adding u->v creates cycle iff v can already reach u
        if nx.has_path(Gt, v, u):
            return False
        Gt.add_edge(u, v)
        return True

    for (a, b) in T:
        data_ab = X[:, [a, b]]  # (n_samples, 2)
        
        if verbose:
            print("Going for orientation")
        
        support, _ = orientation(data_ab, r,verbose = verbose, **orientation_kwargs)  # support is 2x2

        ori = judge_direction(support[0, :], support[1, :])

        na = node_names[a]
        nb = node_names[b]

        if ori == 1:
            print("Phase 2: ori == 1")
            # a -> b
            add_edge_if_no_cycle(na, nb)
        elif ori == -1:
            print("Phase 2: ori == -1")
            # b -> a
            add_edge_if_no_cycle(nb, na)
        else:
            print("Phase 2: ori == 0 (else)")
            # undecided: do nothing (equivalent to "delete undirected edge" for output graph)
            pass

    return Gt, Lalter, T, num_CI_tests



 
