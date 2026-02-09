# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 18:58:19 2025

@author: chdem
"""

from __future__ import annotations

import warnings
from typing import Dict, FrozenSet, Tuple, List
from numpy import ndarray
import numpy as np
import time




from causallearn.graph.Graph import Graph
from causallearn.graph.GeneralGraph import GeneralGraph 

from causallearn.utils.cit import *

import sys, pathlib
repo_root = pathlib.Path(__file__).resolve().parents[3]
sys.path.append(str(repo_root / "src"))

from causaldiscovery.CItest.noCache_CI_Test import myTest
from causaldiscovery.graphs.IncrementalGraph import IncrementalGraph
from causaldiscovery.utils.auxiliary import powerset



num_CI_tests = 0
sepset_size = 0
ci_cache = {}



def ci_pvalue(x: int, y: int, S: FrozenSet[int],  independence_test_method: CIT_Base) -> float:
    global num_CI_tests, sepset_size
    a, b = (x, y) if x <= y else (y, x)
    key = (a, b, S)
    if key in ci_cache:
        return ci_cache[key]
    num_CI_tests += 1
    sepset_size += len(S) 
    p = independence_test_method(a, b, tuple(sorted(S)))
    ci_cache[key] = p
    return p


def mbcs_skeleton(data: ndarray, independence_test_method: CIT_Base, alpha: float = 0.05,  
             verbose: bool = False,  new_node_names:List[str] = None, 
            **kwargs) -> Tuple[Graph, int, int]:
    
    # Initialization
    
   
    
    if type(data) != np.ndarray:
        raise TypeError("'data' must be 'np.ndarray' type!")
    if data.shape[0] < data.shape[1]:
        warnings.warn("The number of features is much larger than the sample size!")

    if type(alpha) != float or alpha <= 0 or alpha >= 1:
        raise TypeError("'alpha' must be 'float' type and between 0 and 1!")
    
    if not isinstance(independence_test_method, CIT_Base) and not isinstance(independence_test_method, myTest):
        raise TypeError("'independence_test_method' must be 'CIT_Base' type!")
        

    
    num_new_vars = data.shape[1]
    
    G = IncrementalGraph(no_of_var = num_new_vars, new_node_names = new_node_names)

    if new_node_names is not None:
        assert len(new_node_names) == num_new_vars, "number of new_node_names: %s  must match number of new variables: %s" % (len(new_node_names), num_new_vars)
        
    for x in range(num_new_vars):
        if verbose:
            print(f"MBCS:{x}/{num_new_vars}")
    
        varS: set[int] = set()
    
        # Grow
        for y in range(num_new_vars):
            if x == y:
                continue
    
            S = frozenset(varS)  # current conditioning set
            p_value = ci_pvalue(x, y, S, independence_test_method)
            # Only count *real* tests (not cache hits)
            # If you want exact counting, track misses inside ci_pvalue instead.
            # For quick approximation:
            # num_CI_tests += 1
            
    
            if p_value < alpha:  # dependent
                if verbose:
                    print(f"Adding {x} o-o {y} with p-value {p_value:.6f}")
                varS.add(y)
    
        # Shrink (iterate over a snapshot because we'll remove)
        for y in list(varS):
            S2 = frozenset(varS - {y})
            p_value = ci_pvalue(x, y, S2, independence_test_method)
    
            if p_value > alpha:
                if verbose:
                    print(f"Removing {x} o-o {y} with p-value {p_value:.6f}")
                varS.remove(y)
    
        for y in varS:
            G.add_edge_with_circles(x, y)
                
    #Collider set search and return
        
    return resolve_markov_blankets_collider_sets(G, independence_test_method, alpha)
        

def resolve_markov_blankets_collider_sets(
    G,
    independence_test_method,
    alpha=0.05
):
    """
    Implements Algorithm 2 (core loop) producing:
      - modified G (spouse links removed at end)
      - C: list of collider orientation directives (x, z, y) meaning x -> z <- y
    """
    spouse_edges = set()  # edges to remove at end (marked spouse links)

    for x, y in G.get_edges_from_triangle():
        Sxy = None

        Bdx = set(G.neighbors(x))
        Bdy = set(G.neighbors(y))
        Trixy = Bdx & Bdy  # nodes that complete triangles with x-y

        # Line 6: B ← smallest set of {Bd(X)\Tri\{Y}, Bd(Y)\Tri\{X}}
        cand1 = Bdx - Trixy - {y}
        cand2 = Bdy - Trixy - {x}
        B = min(cand1, cand2, key=len)

        # for each S ⊆ Tri(X−Y)
        found = False
        for S in powerset(Trixy):
            Z = B | S

            # Line 9: if CONDINDEP(X,Y,Z)
            p = independence_test_method(x, y, tuple(sorted(Z)))
            if p > alpha:
                Sxy = Z
                found = True
                break  # to line 23

         # Line 13: D ← B ∩ {nodes reachable by W in G\XY | W ∈ (Tri \ S)}
            tri_minus_S = Trixy - S
            reachable_union = set()
            
            for w in tri_minus_S:
                reachable_union |= G.reachable_nodes_without_edge(w, x, y)

            D = B & reachable_union
            Bprime = B - D  # Line 14
            
            # Line 15: for each S' ⊆ D
            for Sprime in powerset(D):
                Z2 = Bprime | Sprime | S
                p2 = independence_test_method(x, y, tuple(sorted(Z2)))
                if p2 > alpha:
                    Sxy = Z2
                    found = True
                    break  # to line 23
            if found:
                break

        # Line 23 onwards: save directive if found separator
        if Sxy is not None:
            spouse_edges.add(tuple(sorted((x, y))))  # mark spouse link



    # Line 30: remove all spouse links from G
    for (u, v) in spouse_edges:
        G.remove_if_exists(u, v)

    return G, num_CI_tests, sepset_size