from __future__ import annotations

import warnings
from typing import List, Set, Tuple, Dict
from numpy import ndarray
import numpy as np
import time

from causallearn.graph.Graph import Graph
from causallearn.graph.GeneralGraph import GeneralGraph 
from causallearn.utils.cit import *

from IncrementalGraph import IncrementalGraph
from exists_separator import _exists_separator
import hill_climbing as hc

def s_cdfsf_fs(data: ndarray, independence_test_method: str=fisherz, alpha: float = 0.05,  
              max_iter = 1e4,  initial_graph: GeneralGraph = None,  verbose: bool = False,  new_node_names:List[str] = None, 
               **kwargs) -> Tuple[Graph, int, float, float]:
    
    # Initialization
    
    num_CI_tests = 0
    sepset_size = 0
    initial_time = time.time()
    
    if type(data) != np.ndarray:
        raise TypeError("'data' must be 'np.ndarray' type!")
    
    if data.shape[0] < data.shape[1]:
        warnings.warn("The number of features is much larger than the sample size!")
        
        
    if type(alpha) != float or alpha <= 0 or alpha >= 1:
        raise TypeError("'alpha' must be 'float' type and between 0 and 1!")
        
    if initial_graph is None:
        initial_graph = GeneralGraph([])

    independence_test_method = CIT(data, method=independence_test_method, **kwargs)
    mb = IncrementalGraph(0, initial_graph)
    num_new_vars = data.shape[1] - mb.G.get_num_nodes()

    if new_node_names is not None:
        assert len(new_node_names) == num_new_vars, "number of new_node_names  must match number of new variables"
        
    for t in range(num_new_vars):
        
        if verbose:
            print("Going for var %s of  %s" % (t, num_new_vars))
        
        i = mb.G.get_num_nodes()
        name = None if new_node_names is None else new_node_names[t]
        mb.add_node(name)
        
        # Discovering phase
        for t in range(i):
            mb_t =mb.neighbors(t)
            existis_sep, num_CI, sep_size = _exists_separator(graph = mb, x = t, y = i, mb_x = mb_t, independence_test_method = independence_test_method, alpha = alpha, must_be = (), verbose = verbose)
            num_CI_tests += num_CI
            sepset_size += sep_size
            
            if not existis_sep:
                mb.add_edge_with_circles(t, i)
                
        for t in range(i):
            for y in mb.neighbors(t):
                mb_t = [x for x in mb.neighbors(t) if x != y ]
                existis_sep, num_CI, sep_size = _exists_separator(graph = mb, x= t, y = y, mb_x = mb_t, independence_test_method = independence_test_method, alpha = alpha, must_be = (), verbose = verbose)
                num_CI_tests += num_CI
                sepset_size += sep_size
                if  existis_sep:
                    # note that by construction if Y in CPC(T) then T is in CPC(Y), thus the second if in shrinking phase in the S-CDFSF from Yu, et al. (2012) is unnecessary.
                    mb.remove_if_exists(t, i) 
                    
    mb.G = hc.hill_climbing_search(data, skeleton = mb.G, max_iter = max_iter)    
    
    avg_sepset_size = sepset_size/num_CI_tests
    total_exec_time = time.time() - initial_time
    
    return mb.G, num_CI_tests, avg_sepset_size, total_exec_time
    
"""
Bibliography
    
Yu, K., Wu, X., Ding, W., & Wang, H. (2012). Exploring causal relationships with streaming features. The Computer Journal, 55(9), 1103–1117.
    
    
"""

                    
            