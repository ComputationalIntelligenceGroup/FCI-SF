from causallearn.graph.GeneralGraph import GeneralGraph
from typing import List, Tuple
from itertools import combinations

import numpy as np
import random






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
    
    

        