#!/usr/bin/env python3

import networkx as nx
import pandas as pd

import logging
logging.getLogger('pgmpy').setLevel(logging.WARNING)

from pgmpy.estimators import BICGauss, ExpertKnowledge, HillClimbSearch


from causallearn.graph.GeneralGraph import GeneralGraph 
from causallearn.graph.Endpoint import Endpoint 
from causallearn.graph.Edge import Edge 

from numpy import ndarray

import auxiliary as aux






def hill_climbing_search(dataset: ndarray, skeleton: GeneralGraph,  epsilon: float = 0, verbose: bool = False, tabu_length=0, max_iter = 1e4 ) -> GeneralGraph:


    data = pd.DataFrame(dataset)    
    data.columns = skeleton.get_node_names()
    
    names_map = {data.columns[i]: i for i in range(len(data.columns))}
    
    scoring_method = BICGauss(data)
    hc = HillClimbSearch(data)
    current_edges = set(aux.get_numerical_edges(skeleton))
    
    current_edges.update(set([(j, i) for i, j in current_edges]))
    
   
    
    possible_edges = set(nx.complete_graph(
        n=range(skeleton.get_num_nodes()), create_using=nx.Graph
        ).edges())
    
    possible_edges.update(set([(j, i) for i, j in possible_edges]))
    
    forbidden_edges_num= possible_edges - current_edges
    
    forbidden_edges = set()
    
    for i, j in forbidden_edges_num:
        forbidden_edges.add((data.columns[i], data.columns[j]))
        
    
    expert_knowledge = ExpertKnowledge(
        forbidden_edges = forbidden_edges
        )
    
    model = hc.estimate(
        scoring_method=scoring_method,
        expert_knowledge=expert_knowledge,
        epsilon = epsilon,
        tabu_length=tabu_length,
        max_iter = max_iter,
        show_progress= False
        )    
  
   
    for i, j in [(names_map[name1], names_map[name2]) for (name1, name2) in model.edges()]:
        nodes = skeleton.get_nodes()
        old_edge = skeleton.get_edge(nodes[i], nodes[j])
        if old_edge is None:
            old_edge = skeleton.get_edge(nodes[j], nodes[i])
        skeleton.remove_edge(old_edge)
        new_edge = Edge(nodes[i], nodes[j], Endpoint.TAIL, Endpoint.ARROW)
        skeleton.add_edge(new_edge)
   
    return skeleton
