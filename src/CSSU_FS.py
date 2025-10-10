from __future__ import annotations

import warnings
from typing import List,  Dict, Tuple
from numpy import ndarray
import numpy as np



from causallearn.graph.Graph import Graph
from causallearn.graph.GeneralGraph import GeneralGraph 


from causallearn.utils.cit import *

from IncrementalGraph import IncrementalGraph
import hill_climbing as hc

from SU import gaussian_su_from_samples
import time




def cssu_fs(data: ndarray, alpha: float = 0.1,  
            max_iter = 1e4, initial_graph: GeneralGraph = None,  verbose: bool = False,  new_node_names:List[str] = None, 
            **kwargs) -> Tuple[Graph, int, float, float]:
    
    # Initialization
    num_CI_tests = 0
    initial_time = time.time()
    
    if type(data) != np.ndarray:
        raise TypeError("'data' must be 'np.ndarray' type!")
    if data.shape[0] < data.shape[1]:
        warnings.warn("The number of features is much larger than the sample size!")
        
        
    if initial_graph is None:
        initial_graph = GeneralGraph([])
        
    

    cn = IncrementalGraph(0, initial_graph)
    num_new_vars = data.shape[1] - cn.G.get_num_nodes()

    if new_node_names is not None:
        assert len(new_node_names) == num_new_vars, "number of new_node_names  must match number of new variables"
        
    
    
    for t in range(num_new_vars):
        
        
        j = cn.G.get_num_nodes()
        name = None if new_node_names is None else new_node_names[t]
        cn.add_node(name)
        tempSort: Dict[int , float] = {}
        
        # Step 1: Build candidate neighbors for new feature j by dependece and pseudo-dependece relationship discrimination
        for i in range(j):
            
            num_CI_tests += 1
            
            su = gaussian_su_from_samples(data[:,i], data[:,j])
            if su > alpha:
                
                tempSort[i] = su
                
        while tempSort != {}:
                
            m, su = max(tempSort.items(), key=lambda item: item[1])
            tempSort.pop(m)
            cn.add_edge_with_circles(j, m)

            for  k, v in list(tempSort.items()): 
                num_CI_tests+= 1
                if gaussian_su_from_samples(data[:,k], data[:,m]) > v:
                    if verbose:
                        print("Step 1: Reduce tempSort")
                    tempSort.pop(k)
                    
                    
        # Step 2: Remove psudo-dependence relationships from those neighbors that added the new feature
        
        to_remove = []
        
       
        for m in cn.neighbors(j):
            for s in [x for x in cn.neighbors(m) if x != j]:
                
                num_CI_tests += 2
                if (gaussian_su_from_samples(data[:, j], data[:, m]) > gaussian_su_from_samples(data[:, s], data[:, m])) :
                    num_CI_tests += 2
                    if (gaussian_su_from_samples(data[:, j], data[:, s]) > gaussian_su_from_samples(data[:, s], data[:, m])): # this could be improved by storing the value of the SU of each edge in the graph
                        to_remove.append((m,s))
        
        for m, s in to_remove:
            if verbose:
                print(f"Step 2: Removing {m} : {s}")
            cn.remove_if_exists(m, s)
            
    
    cn.G = hc.hill_climbing_search(data, skeleton = cn.G, max_iter = max_iter)
    total_exec_time = time.time() - initial_time
        
        
    return cn.G, num_CI_tests, 0, total_exec_time
        
"""
Bibliography:
    
Yang, J., Guo, X., An, N., Wang, A., & Yu, K. (2018). Streaming feature-based causal structure learning algorithm with symmetrical uncertainty. Information Sciences, 467, 708–724.


"""
                
                
                
                
        
        
    
    