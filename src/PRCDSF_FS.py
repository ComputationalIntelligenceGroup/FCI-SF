from __future__ import annotations

import warnings
from typing import List, Tuple
from numpy import ndarray
import numpy as np
import time

from causallearn.graph.Graph import Graph
from causallearn.graph.GeneralGraph import GeneralGraph 
from causallearn.utils.cit import *

from IncrementalGraph import IncrementalGraph
import hill_climbing as hc

def prcdsf_fs(data: ndarray, independence_test_method: str=fisherz, alpha: float = 0.05,  
              max_iter = 1e4, initial_graph: GeneralGraph = None,  verbose: bool = False,  new_node_names:List[str] = None,
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
            print(f"PRCDSF: {t}/{num_new_vars}")
            
        j = mb.G.get_num_nodes()
        name = None if new_node_names is None else new_node_names[t]
        mb.add_node(name)
        
        #Relevance analysis
        for i in range(j):
            
            S = tuple() if mb.neighbors(i).size == 0 else tuple(mb.neighbors(i))
            
            num_CI_tests += 1
            sepset_size += len(S)
            
            p_value = independence_test_method(j, i, S)
            if p_value < alpha:
                if verbose:
                    print('%s not ind %s  with p-value %f\n' % (j, i, p_value))
                mb.add_edge_with_circles(i, j)
                
         #Redundancy analysis
         
        for i in mb.neighbors(j):
            for k in [x for x in mb.neighbors(i) if x != j]: # Quitar la j
                
                S = tuple((x for x in mb.neighbors(i) if x != k))
                
                num_CI_tests += 1
                sepset_size += len(S)
                
                p_value = independence_test_method(i, k, S)
                if p_value >= alpha:
                    if verbose:
                        print('%s ind %s  with p-value %f\n' % (j, i, p_value))
                    mb.remove_if_exists(i, k)
    
                    
    if verbose:
        print("PRCDSF: hill-climbing")
        
    mb.G = hc.hill_climbing_search(data, skeleton = mb.G, max_iter = max_iter)
    
    avg_sepset_size = sepset_size/num_CI_tests
    total_exec_time = time.time() - initial_time
    
    return mb.G, num_CI_tests, avg_sepset_size, total_exec_time

"""
Bibliography

Yang, J., Jiang, L., Shen, A., & Wang, A. (2022). Online streaming features causal discovery algorithm based on partial rank correlation. IEEE Transactions on Artificial Intelligence, 4(1), 197–208.

"""
            
            
        
        