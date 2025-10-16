from __future__ import annotations

 

from causallearn.utils.DepthChoiceGenerator import DepthChoiceGenerator
from typing import List, Set, Tuple, Dict
from causallearn.utils.cit import *

from IncrementalGraph import IncrementalGraph

def _exists_separator(graph: IncrementalGraph, x:int, y:int,  mb_x: List[int], independence_test_method: CIT_Base, alpha: float = 0.05,  must_be:Tuple[int] = None, max_sepset_size: int = -1, verbose: bool = True):
    
    num_CI = 0
    sepset_size = 0
    
    if must_be is None:
        must_be = ()
    
    max_len = max_sepset_size if max_sepset_size >= 0 else float('inf')
    
    m_aux = [aux_var for aux_var in mb_x if aux_var != x and aux_var != y ]
    gen = DepthChoiceGenerator(len(m_aux), len(m_aux))
    choice = gen.next()
    exists_separator = False
    
    last_len = 0
    
    while choice is not None and not exists_separator:
        
        S = tuple([m_aux[index] for index in choice]) + tuple(must_be)
        
        if len(S) - len(must_be) > max_len:
            break
        
        num_CI += 1
        sepset_size += len(S)
        
        p_value = independence_test_method(x, y, S)
        exists_separator = p_value > alpha
        
        if verbose and last_len < len(S):
            last_len = len(S)
            print(f"Sepset len: {last_len}")
        if exists_separator and verbose:
            print("%s ind %s given %s with p-val %s" % (x, y, S, p_value))
        choice = gen.next()
        
    return exists_separator, num_CI, sepset_size