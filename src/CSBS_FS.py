from __future__ import annotations

import warnings
from typing import List,  Tuple
from numpy import ndarray
import numpy as np
import time



from causallearn.graph.Graph import Graph
from causallearn.graph.GeneralGraph import GeneralGraph 

from causallearn.utils.cit import *

from IncrementalGraph import IncrementalGraph
from exists_separator import _exists_separator
import hill_climbing as hc


def csbs_fs(data: ndarray, independence_test_method: str=fisherz, alpha: float = 0.05,  
            max_iter = 1e4, initial_graph: GeneralGraph = None,  verbose: bool = False,  new_node_names:List[str] = None, 
            **kwargs) -> Tuple[Graph, int, float, float]:
    
    # Initialization
    
    print("Starting csbs")
    
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
        assert len(new_node_names) == num_new_vars, "number of new_node_names: %s  must match number of new variables: %s" % (len(new_node_names), num_new_vars)
        
    for t in range(num_new_vars):
        
        if verbose:
            print(f"CSBS:{t}/{num_new_vars}")
        tempmb: List[int] = []
        j = mb.G.get_num_nodes()
        name = None if new_node_names is None else new_node_names[t]
        mb.add_node(name)
        
        # Relevance analysis
        for i in range(j):
            num_CI_tests += 1
            p_value = independence_test_method(j, i, ())
            if p_value < alpha:
                if verbose:
                    print('Adding %s o-o %s  with p-value %f\n' % (j, i, p_value))
                tempmb.append(i)
            
        # Redundancy analysis
        for k in tempmb:
            mb_k =mb.neighbors(k).copy()
            
            exist_sep, num_CI, seps_size = _exists_separator(mb, k, j, mb_k, independence_test_method, alpha, (), verbose = verbose)
            num_CI_tests += num_CI
            sepset_size += seps_size
            
            if not exist_sep:
                mb.add_edge_with_circles(k, j)

                for s in mb_k: #mb_k does not cointain j
                    exist_sep, num_CI, seps_size = _exists_separator(mb, k, s, mb_k, independence_test_method, alpha, (j, ), verbose = verbose)
                    num_CI_tests += num_CI
                    sepset_size += seps_size
                    if exist_sep:
                        mb.remove_if_exists(k, s)
                    
        # redundancy analysis for MB{j}
        to_remove = []
        for h in mb.neighbors(j):
            exist_sep, num_CI, seps_size = _exists_separator(mb, j, h, mb.neighbors(j), independence_test_method, alpha, (), verbose = verbose)
            num_CI_tests += num_CI
            sepset_size += seps_size
            if  exist_sep:
                to_remove.append((j, h))
            
        for j_r, h_r in to_remove:
            mb.remove_if_exists(j_r, h_r)
    
    if verbose:
        print("CSBS: hill-climbing")
    mb.G = hc.hill_climbing_search(data, skeleton = mb.G, max_iter = max_iter)
        
    avg_sepset_size = sepset_size/num_CI_tests
    total_exec_time = time.time() - initial_time
    
    return mb.G, num_CI_tests, avg_sepset_size, total_exec_time
            
"""
Bibliography

Guo, X., & Yang, J. (2017). Causal structure learning algorithm based on streaming features. 2017 IEEE International Conference on Big Knowledge (ICBK), 192–197. https://ieeexplore.ieee.org/abstract/document/8023415/

"""    
                    
                
               
                
                
   


   