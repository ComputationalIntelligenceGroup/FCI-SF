#!/usr/bin/env python3

import networkx as nx
import pandas as pd

import logging
logging.getLogger('pgmpy').setLevel(logging.WARNING)

from pgmpy.estimators import BIC, ExpertKnowledge, HillClimbSearch


from causallearn.graph.GeneralGraph import GeneralGraph 
from causallearn.graph.Endpoint import Endpoint 
from causallearn.graph.Edge import Edge 

from numpy import ndarray

import auxiliary as aux






def hill_climbing_search(dataset: ndarray, skeleton: GeneralGraph, epsilon: float = 0, verbose: bool = False, tabu_length=0, max_iter = 1e4 ) -> GeneralGraph:


    data = pd.DataFrame(dataset)    
    scoring_method = BIC(data)
    hc = HillClimbSearch(data)
    current_edges = set(aux.get_numerical_edges(skeleton))
    
    current_edges.update(set([(j, i) for i, j in current_edges]))
    
   
    
    possible_edges = set(nx.complete_graph(
        n=range(skeleton.get_num_nodes()), create_using=nx.Graph
        ).edges())
    
    possible_edges.update(set([(j, i) for i, j in possible_edges]))
    
    expert_knowledge = ExpertKnowledge(
        forbidden_edges= possible_edges - current_edges
        )
    
    model = hc.estimate(
        scoring_method=scoring_method,
        expert_knowledge=expert_knowledge,
        epsilon = epsilon,
        tabu_length=tabu_length,
        max_iter = max_iter,
        show_progress= False
        )
    

  
   
    for i, j in model.edges():
        nodes = skeleton.get_nodes()
        old_edge = skeleton.get_edge(nodes[i], nodes[j])
        skeleton.remove_edge(old_edge)
        new_edge = Edge(nodes[i], nodes[j], Endpoint.TAIL, Endpoint.ARROW)
        skeleton.add_edge(new_edge)
   
    return skeleton
