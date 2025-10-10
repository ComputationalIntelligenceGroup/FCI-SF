from __future__ import annotations


from typing import List, Dict, Tuple, Set
from numpy import ndarray
from itertools import chain, combinations
import numpy as np

from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.utils.cit import *


from IncrementalGraph import IncrementalGraph





def fas_fs(data: ndarray,  independence_test_method: CIT_Base, alpha: float = 0.05, 
           initial_sep_sets: Dict[Tuple[int, int], Set[int]] = None, initial_graph: GeneralGraph = None, 
           depth: int = -1, verbose: bool = False, stable: bool = True, new_node_names: List[str] = None) -> Tuple[
    GeneralGraph, Dict[Tuple[int, int], Set[int]], int, int]:
    """
    Implements the "fast adjacency search" used in several causal algorithm in this file. In the fast adjacency
    search, at a given stage of the search, an edge X*-*Y is removed from the graph if X _||_ Y | S, where S is a subset
    of size d either of adj(X) or of adj(Y), where d is the depth of the search. The fast adjacency search performs this
    procedure for each pair of adjacent edges in the graph and for each depth d = 0, 1, 2, ..., d1, where d1 is either
    the maximum depth or else the first such depth at which no edges can be removed. The interpretation of this adjacency
    search is different for different algorithm, depending on the assumptions of the algorithm. A mapping from {x, y} to
    S({x, y}) is returned for edges x *-* y that have been removed.

    Parameters
    ----------
    data: data set (numpy ndarray), shape (n_samples, n_features). The input data, where n_samples is the number of
            samples and n_features is the number of features.
    new_nodes: The new search nodes.
    initial_sep_sets: The initial_sep_sets from the previous iterations of FAS-FS
    independence_test_method: the function of the independence test being used
            [fisherz, chisq, gsq, kci]
           - fisherz: Fisher's Z conditional independence test
           - chisq: Chi-squared conditional independence test
           - gsq: G-squared conditional independence test
           - kci: Kernel-based conditional independence test
    alpha: float, desired significance level of independence tests (p_value) in (0,1)
    knowledge: background background_knowledge
    depth: the depth for the fast adjacency search, or -1 if unlimited
    verbose: True is verbose output should be printed or logged
    stable: run stabilized skeleton discovery if True (default = True)
    Returns
    -------
    graph: Causal graph skeleton, where graph.graph[i,j] = graph.graph[j,i] = -1 indicates i --- j.
    initial_sep_sets: Separated sets of graph
    num_CI: Number of performed CI tests
    sep_size: Sum of the size of all the conditioning sets that were used in each CI test
    """
    ## ------- check parameters ------------
    if type(data) != np.ndarray:
        raise TypeError("'data' must be 'np.ndarray' type!")
    if not isinstance(independence_test_method, CIT_Base):
        raise TypeError("'independence_test_method' must be 'CIT_Base' type!")
    if type(alpha) != float or alpha <= 0 or alpha >= 1:
        raise TypeError("'alpha' must be 'float' type and between 0 and 1!")
    if type(depth) != int or depth < -1:
        raise TypeError("'depth' must be 'int' type >= -1!")
    ## ------- end check parameters ------------

    num_CI = 0
    sep_size = 0
    

    if depth == -1:
        depth = float('inf')
        
    if initial_graph is None:
        
        initial_graph = GeneralGraph([])
        
    if initial_sep_sets is None:
        initial_sep_sets = {}
        
   
    # Initialize initial values and structures.
    
    num_old_vars = initial_graph.get_num_nodes()
    num_new_vars = data.shape[1] - num_old_vars
    

    
    
    
    sep_sets = initial_sep_sets
    
    
    
    inc_graph = IncrementalGraph( num_new_vars, initial_graph, new_node_names = new_node_names)
    
    inc_graph.initial_skeleton()
    
    
    
    current_depth: int = -1
    
    
    
    while inc_graph.max_degree() > current_depth and current_depth < depth:
        current_depth += 1
        
        if stable:
            neighbors = {node_num: set(inc_graph.neighbors(node_num)) for node_num in range(inc_graph.G.get_num_nodes())}
           
            
        for x, y in inc_graph.get_numerical_edges():
            
           
            if stable: 
                neigh_x, neigh_y = neighbors[x], neighbors[y]
            else:
                neigh_x, neigh_y = inc_graph.neighbors(x), inc_graph.neighbors(y)
            
            for separating_set in chain(
                combinations(set(neigh_x) - set([y]), current_depth),
                combinations(set(neigh_y) - set([x]), current_depth),
            ):
                
                
                #Skip repeating tests from previous iterations
                if x < num_old_vars and y < num_old_vars and all (z < num_old_vars for z in separating_set):
                    continue
                    
               
                
                # If a conditioning set exists remove the edge, store the
                # separating set and move on to finding conditioning set for next edge.
                num_CI += 1
                sep_size += len(separating_set)
                
                p = independence_test_method(x, y, separating_set)
                
                
                if p > alpha:
                    if verbose:
                       print('%s ind %s | %s with p-value %f\n' % (x, y, separating_set, p))
                        
                    inc_graph.remove_if_exists(x, y)
                    inc_graph.remove_if_exists(y, x)
                    sep_sets[(x, y)] = set(separating_set)
                    sep_sets[(y, x)] = set(separating_set)
                    
                    break
                    
                
    return initial_graph, sep_sets, num_CI, sep_size
    
    
