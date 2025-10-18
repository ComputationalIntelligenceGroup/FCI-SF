#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 17:29:59 2025

@author: chema
"""

from typing import List, Dict

import numpy as np
import networkx as nx


from causallearn.graph.GeneralGraph import GeneralGraph
import causallearn.search.ConstraintBased.FCI as fci
from causallearn.graph.Endpoint import Endpoint

from causaldag import DAG

from FCI_FS import fci_fs, removeByPossibleDsep
from IncrementalGraph import IncrementalGraph

from causallearn.utils.cit import CIT_Base, NO_SPECIFIED_PARAMETERS_MSG
from collections import Counter


def dag2pag(ground_truth_DAG, obsVars: List[str], new = False) -> GeneralGraph:
    
    
    data = np.empty(shape=(0, len(obsVars)))
    
    independence_test_method = D_Sep(data, name_index_mapping = obsVars, true_dag=ground_truth_DAG)
    
    if not new:
        output_fci = fci_fs(data, independence_test_method=independence_test_method, initial_sep_sets = {}, initial_graph = GeneralGraph([]), new_node_names = obsVars, verbose = False)
        
        print(f"Exec time: {output_fci[3]}")
        
        return output_fci[0]
    
    else: 
        dag = DAG(arcs=set(ground_truth_DAG.edges()))

        # 'L' es una variable latente
        name_num = 0
        name_map = {}
        for node_name in ground_truth_DAG.nodes:
            name_map[node_name] = name_num
            name_num += 1
            
        latent_nodes = set(ground_truth_DAG.nodes) - set(obsVars)
        
        mag = dag.marginal_mag(latent_nodes)
        
        dir_coll, bidir_coll, undir_edges = get_colliders_mag(mag)
        
        
            
        
        inc_graph = IncrementalGraph( len(obsVars), IncrementalGraph([]), new_node_names = obsVars)
        
        for a, b in dir_coll:
            
            inc_graph.add_edge_directed(name_map[a], name_map[b])
            
        for a, b in bidir_coll:
            inc_graph.add_edge_bidirected(name_map[a], name_map[b])
            
        for a, b in undir_edges:
            inc_graph.add_edge_with_circles(name_map[a], name_map[b])
            
            
        
        """
        
        graph = inc_graph.G
        
        num_CI, sep_size = removeByPossibleDsep(graph, independence_test_method, 0.5, {}, [])
        

        fci.reorientAllWith(graph, Endpoint.CIRCLE)
        fci.rule0(graph, nodes, sep_sets, None, verbose)

        change_flag = True
        

        while change_flag:
            change_flag = False
            change_flag = fci.rulesR1R2cycle(graph, None, change_flag, verbose)
            change_flag = fci.ruleR3(graph, sep_sets, None, change_flag, verbose)

            if change_flag:
                change_flag = fci.ruleR4B(graph, max_path_length, dataset, independence_test_method, alpha, sep_sets,
                                      change_flag,
                                      None, verbose)

                
                if verbose:
                    print("Epoch")

            # rule 5
            change_flag = fci.ruleR5(graph, change_flag, verbose)
            
            # rule 6
            change_flag = fci.ruleR6(graph, change_flag, verbose)
            
            # rule 7
            change_flag = fci.ruleR7(graph, change_flag, verbose)
            
            # rule 8
            change_flag = fci.rule8(graph,nodes, change_flag)
            
            # rule 9
            change_flag = fci.rule9(graph, nodes, change_flag)
            # rule 10
            change_flag = fci.rule10(graph, change_flag)

        graph.set_pag(True)
        
        """
        
        
def filter_second_repeated(s):

    
    cuenta = Counter(b for _, b in s)

    return {t for t in s if cuenta[t[1]] > 1}


def filter_any_repeated(s):

   
    count = Counter(x for t in s for x in t)
    
    return {t for t in s if count[t[0]] > 1 or count[t[1]] > 1}

def find_coincidence(A, B):
   
    B_vals = {x for t in B for x in t}

    A_coincidence = set()
    B_coincidence = set()

    for a in A:
        segundo = a[1]
        if segundo in B_vals:
            A_coincidence.add(a)
            # añadimos todas las tuplas de B donde aparece ese valor
            B_coincidence |= {b for b in B if segundo in b}

    return A_coincidence, B_coincidence


def get_colliders_mag(mag):
    
    bidir_edges = set(tuple(sorted(edge)) for edge in mag.bidirected)
    
    dir_coll, bidir_coll = find_coincidence(mag.directed, bidir_edges)
    
    dir_coll |= filter_second_repeated(mag.directed)
    
    bidir_coll |= filter_any_repeated(bidir_edges)
    
    undir_edges = (mag.directed - dir_coll) | (bidir_edges - bidir_coll)
    
    return dir_coll, bidir_coll, undir_edges
   
            


class D_Sep(CIT_Base):
    def __init__(self, data,  name_index_mapping: Dict[str, int], true_dag, **kwargs):
        '''
        Use d-separation as CI test, to ensure the correctness of constraint-based methods. (only used for tests)
        Parameters
        ----------
        data:   numpy.ndarray, just a placeholder, not used in D_Separation
        true_dag:   nx.DiGraph object, the true DAG
        '''
        super().__init__(data, **kwargs)  # data is just a placeholder, not used in D_Separation
        self.check_cache_method_consistent('d_separation', NO_SPECIFIED_PARAMETERS_MSG)
        self.true_dag = true_dag
        self.name_index_mapping = name_index_mapping
        
        # import networkx here violates PEP8; but we want to prevent unnecessary import at the top (it's only used here)

    def __call__(self, X, Y, condition_set=None):
        Xs, Ys, condition_set, cache_key = self.get_formatted_XYZ_and_cachekey(X, Y, condition_set)
        if cache_key in self.pvalue_cache: return self.pvalue_cache[cache_key]
        p = float(nx.is_d_separator(self.true_dag, {self.name_index_mapping[Xs[0]]}, {self.name_index_mapping[Ys[0]]}, {self.name_index_mapping[z] for z in condition_set}))
        # pvalue is bool here: 1 if is_d_separated and 0 otherwise. So heuristic comparison-based uc_rules will not work.

        # here we use networkx's d_separation implementation.
        # an alternative is to use causal-learn's own d_separation implementation in graph class:
        #   self.true_dag.is_dseparated_from(
        #       self.true_dag.nodes[Xs[0]], self.true_dag.nodes[Ys[0]], [self.true_dag.nodes[_] for _ in condition_set])
        #   where self.true_dag is an instance of GeneralGrpah class.
        # I have checked the two implementations: they are equivalent (when the graph is DAG),
        # and generally causal-learn's implementation is faster.
        # but just for now, I still use networkx's, for two reasons:
        # 1. causal-learn's implementation sometimes stops working during run (haven't check detailed reasons)
        # 2. GeneralGraph class will be hugely refactored in the near future.
        self.pvalue_cache[cache_key] = p
        return p