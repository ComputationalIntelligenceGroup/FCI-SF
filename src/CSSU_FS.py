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

import numpy as np
from pyitlib import discrete_random_variable as drv


from skfeature.utility.mutual_information import su_calculation




def cssu_fs(data: ndarray, alpha: float = 0.1,  initial_graph: GeneralGraph = GeneralGraph([]),  verbose: bool = False,  new_node_names:List[str] = None, **kwargs) -> Graph:
    
    # Initialization
    
    if type(data) != np.ndarray:
        raise TypeError("'data' must be 'np.ndarray' type!")
    
    if data.shape[0] < data.shape[1]:
        warnings.warn("The number of features is much larger than the sample size!")
        
        
    if type(alpha) != float or alpha <= 0 or alpha >= 1:
        raise TypeError("'alpha' must be 'float' type and between 0 and 1!")

   
    cn = IncrementalGraph(0, initial_graph)
    num_new_vars = data.shape[1] - cn.G.get_num_nodes()

    if new_node_names is not None:
        assert len(new_node_names) == num_new_vars, "number of new_node_names  must match number of new variables"
        
    
    
    for t in range(num_new_vars):
        
        if verbose:
            print("Going for var %s of  %s" % (t, num_new_vars))
        
        j = cn.G.get_num_nodes()
        name = None if new_node_names is None else new_node_names[t]
        
        cn.add_node(name)
        
        tempSort: Dict[int , float] = {}
        
        # Step 1: Build candidate neighbors for new feature j by dependece and pseudo-dependece relationship discrimination
        
        for i in range(j):
            
            
            su = su_calculation(data[:,i], data[:,j] )
            
            if su > alpha:
                
                tempSort[i] = su
                
        while tempSort != {}:
                
            m, su = max(tempSort.items(), key=lambda item: item[1])
                
            tempSort.pop(m)
                
            cn.add_edge_with_circles(j, m)
                
            for k in tempSort.keys():
                    
                if su_calculation(data[:,k], data[:,m]) > tempSort[k]:
    
                    tempSort.pop(k)
                    
                    
        # Step 2: Remove psudo-dependence relationships from those neighbors has added the new feature
        
        if verbose:
            print("Step 2: neighborhood of %s : %s" % (j, cn.neighbors(j)))
        for m in cn.neighbors(j):
            for s in [x for x in cn.neighbors(m) if x != j]:
                if (su_calculation(data[:, j], data[:, m]) > su_calculation(data[:, s], data[:, m])) and (su_calculation(data[:, j], data[:, s]) > su_calculation(data[:, s], data[:, m])): # this could be improved by storing the value of the SU of each edge in the graph
                    cn.remove_if_exists(m, s)
        
        
    return cn.G
        
"""
Bibliography:
    
Yang, J., Guo, X., An, N., Wang, A., & Yu, K. (2018). Streaming feature-based causal structure learning algorithm with symmetrical uncertainty. Information Sciences, 467, 708–724.


"""
                
                
                
                
        
        
    
    