#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 17:29:59 2025

@author: chema
"""
from itertools import combinations, permutations
from typing import List, Dict

import numpy as np
import networkx as nx
from networkx.algorithms import d_separated

from causallearn.graph.Dag import Dag
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.Node import Node
from causallearn.search.ConstraintBased.FCI import rule0, rulesR1R2cycle, ruleR3, ruleR4B, ruleR5, ruleR6, ruleR7, rule8, rule9, rule10
from causallearn.utils.cit import CIT, d_separation
from FCI_FS import fci_fs

from causallearn.utils.cit import CIT_Base, NO_SPECIFIED_PARAMETERS_MSG


def dag2pag(ground_truth_DAG, obsVars: List[str]) -> GeneralGraph:
    
     
    data = np.empty(shape=(0, len(obsVars)))
    
    output_fci = fci_fs(data, independence_test_method=D_Sep(data, name_index_mapping = obsVars, true_dag=ground_truth_DAG), initial_sep_sets = {}, initial_graph = GeneralGraph([]), new_node_names = obsVars, verbose = False)
    
    return output_fci[0]


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