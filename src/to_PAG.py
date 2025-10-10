#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from IncrementalGraph import IncrementalGraph
from typing import List

from src.causal_graphs.pag import PAG

def to_PAG(model, names:List[str]):
    
    no_vars = len(names)
    
    names_map = {names[i]: i for i in range(no_vars)}

    graph = IncrementalGraph(no_of_var = no_vars, new_node_names = names)
    nodes = graph.G.get_nodes()
    
    for i, j in [(names_map[name1], names_map[name2]) for (name1, name2) in model.edges()]:
        graph.G.add_directed_edge(nodes[i], nodes[j])
        
    return PAG(graph.G)