from __future__ import annotations

import warnings
from typing import List, Set, Tuple, Dict
from numpy import ndarray
import numpy as np



from causallearn.graph.Graph import Graph
from causallearn.graph.GeneralGraph import GeneralGraph 


from causallearn.utils.cit import *

from IncrementalGraph import IncrementalGraph
from exists_separator import _exists_separator



    


def csbs_fs(data: ndarray, independence_test_method: str=fisherz, alpha: float = 0.05,  initial_graph: GeneralGraph = GeneralGraph([]),  verbose: bool = False,  new_node_names:List[str] = None, **kwargs) -> Graph:
    
    # Initialization
    
    if type(data) != np.ndarray:
        raise TypeError("'data' must be 'np.ndarray' type!")
    
    if data.shape[0] < data.shape[1]:
        warnings.warn("The number of features is much larger than the sample size!")
        
        
    if type(alpha) != float or alpha <= 0 or alpha >= 1:
        raise TypeError("'alpha' must be 'float' type and between 0 and 1!")

    independence_test_method = CIT(data, method=independence_test_method, **kwargs)
    mb = IncrementalGraph(0, initial_graph)
    num_new_vars = data.shape[1] - mb.G.get_num_nodes()

    if new_node_names is not None:
        assert len(new_node_names) == num_new_vars, "number of new_node_names  must match number of new variables"
        
    for t in range(num_new_vars):
        tempmb: List[int] = []
        j = mb.G.get_num_nodes()
        name = None if new_node_names is None else new_node_names[t]
        mb.add_node(name)
        
        # Relevance analysis
        for i in range(j):
            p_value = independence_test_method(j, i, ())
            if p_value < alpha:
                if verbose:
                    print('Adding %s o-o %s  with p-value %f\n' % (j, i, p_value))
                tempmb.append(i)
            
        # Redundancy analysis
        for k in tempmb:
            mb_k =mb.neighbors(k).copy()
            if not _exists_separator(mb, k, j, mb_k, independence_test_method, alpha, (), verbose = verbose):
                mb.add_edge_with_circles(k, j)

                for s in mb_k: #mb_k does not cointain j
                    if _exists_separator(mb, k, s, mb_k, independence_test_method, alpha, (j, ), verbose = verbose):
                        mb.remove_if_exists(k, s)
                    
        # redundancy analysis for MB{j}
        for h in mb.neighbors(j):
            if  _exists_separator(mb, j, h, mb.neighbors(j), independence_test_method, alpha, (), verbose = verbose):
                mb.remove_if_exists(j, h)

    return mb.G
            
"""
Bibliography

Guo, X., & Yang, J. (2017). Causal structure learning algorithm based on streaming features. 2017 IEEE International Conference on Big Knowledge (ICBK), 192–197. https://ieeexplore.ieee.org/abstract/document/8023415/

"""    
                    
                
               
                
                
   


   