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
    

    output_fci = fci_fs(data, independence_test_method=independence_test_method, initial_sep_sets = {}, initial_graph = GeneralGraph([]), new_node_names = obsVars, verbose = False)
        
       
        
    return output_fci[0]
    
    
        
        
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
        self.true_dag = true_dag
        self.name_index_mapping = name_index_mapping
        
        # import networkx here violates PEP8; but we want to prevent unnecessary import at the top (it's only used here)

    def __call__(self, X, Y, condition_set=None):
        
    
        # pvalue is bool here: 1 if is_d_separated and 0 otherwise. So heuristic comparison-based uc_rules will not work.

        return float(nx.is_d_separator(self.true_dag, {self.name_index_mapping[X]}, {self.name_index_mapping[Y]}, {self.name_index_mapping[z] for z in condition_set}))