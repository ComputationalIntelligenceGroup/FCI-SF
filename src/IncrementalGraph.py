from __future__ import annotations


from typing import List, Tuple, Set



from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.Node import Node
import numpy as np

import auxiliary as aux




class IncrementalGraph:
    
    def __init__(self, no_of_var: int, initial_graph: GeneralGraph = None,  new_node_names: List[str] = None):
        
        if initial_graph is None:
            initial_graph = GeneralGraph([]) 
        
        if new_node_names is None:
            new_node_names: List[str] = []
           
            for i in range(initial_graph.get_num_nodes(), initial_graph.get_num_nodes() + no_of_var):
                name = "X%d" % (i+1)
                new_node_names.append(name)
                
        else:
            assert len(new_node_names) == no_of_var, "number of new_node_names %s  must match number of variables %s " % (len(new_node_names), no_of_var)
        
        
        assert (len(new_node_names) + initial_graph.get_num_nodes()) == len(set(new_node_names).union(set(initial_graph.get_node_names()))), "Every node in the graph must have a unique name"

        new_nodes: List[Node] = []

        
        for i in range(no_of_var):         
            node = GraphNode(new_node_names[i])
            id: int = i + initial_graph.get_num_nodes()
            node.add_attribute("id", id )
            new_nodes.append(node)
           
        self.old_nodes = initial_graph.get_nodes().copy()
        
        self.new_nodes = new_nodes
        
        
        for new_node in self.new_nodes: 
            initial_graph.add_node(new_node)
            
        self.G = initial_graph
        
        self.undirected()
        
        
        
       
    
    def add_node(self, node_name: str = None):
        
        if node_name is None:
            name = "X%d" % (self.G.get_num_nodes() + 1)
        else: 
            name = node_name
            
        node = GraphNode(name)
        id: int = self.G.get_num_nodes()
        node.add_attribute("id", id )
        
        self.old_nodes.extend(self.new_nodes)
        self.new_nodes = [node]
        self.G.add_node(node)
        
    
    def add_edge_with_circles(self, i: int, j: int):
        
        nodes = self.G.get_nodes()
        self.G.add_edge(Edge(nodes[i], nodes[j], Endpoint.CIRCLE, Endpoint.CIRCLE))
        
        
        
        
       
        
        
    
    
    def undirected(self) -> None:
        """Eliminate all the directions of the graph, leaving it's skeleton."""
        
        adj: Set[Tuple[Node, Node]] = set()
        for edge in self.G.get_graph_edges():
            adj.add((edge.get_node1(), edge.get_node2()))
            self.G.remove_edge(edge)
            
        
        for node1, node2 in adj:
            edge = Edge(node1, node2, Endpoint.CIRCLE, Endpoint.CIRCLE)
            self.G.add_edge(edge)

    def initial_skeleton(self) -> None:
        """Create the initial skeleton by adding all the new variables in the graph"""
        
        self.undirected()
        
        for new_node in self.new_nodes: 
            for old_node in self.old_nodes:
                new_edge = Edge(new_node, old_node , Endpoint.CIRCLE, Endpoint.CIRCLE)
                self.G.add_edge(new_edge)
                
        for i in range(len(self.new_nodes)):
            for j in range(i+1, len(self.new_nodes)):
                #Create complete graph with new nodes
                new_edge = Edge(self.new_nodes[i], self.new_nodes[j] , Endpoint.CIRCLE, Endpoint.CIRCLE)
                self.G.add_edge(new_edge)
                
                
    def neighbors(self, i: int) -> np.ndarray:
        """Find the neighbors of node i in adjmat"""
        return np.where(self.G.graph[i, :] != 0)[0]
    
    
        
    
    def max_degree(self) -> int:
        """Return the maximum number of edges connected to a node in adjmat"""
        return max(np.sum(self.G.graph != 0, axis=1)) - 1
    
    def get_numerical_edges(self) -> List[Tuple[int, int]]:
        """Returns all the edges from the incremental graph"""
        
        return aux.get_numerical_edges(self.G)
            
    def remove_if_exists(self, x: int, y: int) -> None:
        edge = self.G.get_edge(self.G.nodes[x], self.G.nodes[y])
        if edge is not None:
            self.G.remove_edge(edge)
        
    
                
                
    
    
    
    
            
            
            
        
        

        
     


    