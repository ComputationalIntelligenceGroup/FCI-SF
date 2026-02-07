# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 18:58:19 2025

@author: chdem
"""

from __future__ import annotations

import warnings
from typing import List,  Tuple
from numpy import ndarray
import numpy as np
import time



from causallearn.graph.Graph import Graph
from causallearn.graph.GeneralGraph import GeneralGraph 

from causallearn.utils.cit import *

from noCache_CI_Test import myTest

from IncrementalGraph import IncrementalGraph
from exists_separator import _exists_separator
import hill_climbing as hc


def mbcs(data: ndarray, independence_test_method: CIT_Base, alpha: float = 0.05,  
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
    if not isinstance(independence_test_method, CIT_Base) and not isinstance(independence_test_method, myTest):
        raise TypeError("'independence_test_method' must be 'CIT_Base' type!")
        

    mb = IncrementalGraph(0, initial_graph)
    num_new_vars = data.shape[1] - mb.G.get_num_nodes()

    if new_node_names is not None:
        assert len(new_node_names) == num_new_vars, "number of new_node_names: %s  must match number of new variables: %s" % (len(new_node_names), num_new_vars)
        
    for t in range(num_new_vars):
        
        if verbose:
            print(f"MBCS:{t}/{num_new_vars}")
        temp_S: List[int] = []
        j = mb.G.get_num_nodes()
        name = None if new_node_names is None else new_node_names[t]
        mb.add_node(name)
        
        # Grow
        for i in range(num_new_vars):
            
            if i == t:
                continue
            
            num_CI_tests += 1
            p_value = independence_test_method(j, i, tuple(temp_S))
            if p_value < alpha:
                if verbose:
                    print('Adding %s o-o %s  with p-value %f\n' % (j, i, p_value))
                temp_S.append(i)
                
        # Shrink
        for i in temp_S:
            S2 = [s for s in temp_S if s != i]
            
            num_CI_tests += 1
            p_value = independence_test_method(j, i, tuple(S2))
            if p_value > alpha:
                if verbose:
                    print('Removing %s o-o %s  with p-value %f\n' % (j, i, p_value))
                temp_S.remove(i)
        #TODO: Add link
        
        
    #Collider set search
    C = []