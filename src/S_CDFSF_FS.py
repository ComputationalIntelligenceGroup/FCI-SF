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

def s_cdfsf_fs(data: ndarray, independence_test_method: str=fisherz, alpha: float = 0.05,  initial_graph: GeneralGraph = GeneralGraph([]),  verbose: bool = False,  new_node_names:List[str] = None, **kwargs) -> Graph:
    
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
        
        if verbose:
            print("Going for var %s of  %s" % (t, num_new_vars))
        
        i = mb.G.get_num_nodes()
        name = None if new_node_names is None else new_node_names[t]
        mb.add_node(name)
        
        # Discovering phase
        for t in range(i):
            mb_t =mb.neighbors(t)
            if not _exists_separator(mb, t, i, mb_t, independence_test_method, alpha, ()):
                mb.add_edge_with_circles(t, i)
                
        for t in range(i):
            for y in mb.neighbors(t):
                mb_t = [x for x in mb.neighbors(t) if x != y ]
                if  _exists_separator(mb, t, y, mb_t, independence_test_method, alpha, ()):
                    # note that by construction if Y in CPC(T) then T is in CPC(Y), thus the second if in shrinking phase in the S-CDFSF from Yu, et al. (2012) is unnecessary.
                    mb.remove_if_exists(t, i) 
                    
        
    return mb.G
    
"""
Bibliography
    
Yu, K., Wu, X., Ding, W., & Wang, H. (2012). Exploring causal relationships with streaming features. The Computer Journal, 55(9), 1103–1117.
    
    
"""

                    
            