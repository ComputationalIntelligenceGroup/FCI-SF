#!/usr/bin/env python3



import sys, pathlib
repo_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root / "src" / "external" / "causal-self-compatibility" / "src" ))

from causal_graphs.pag import PAG
from self_compatibility import SelfCompatibilityScorer    # if you need the scorer

from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.Node import Node
from causallearn.utils.GraphUtils    import GraphUtils

import graphical_metrics



def add_bidirected_edge(graph: GeneralGraph, node1: Node, node2: Node):
    i = graph.node_map[node1]
    j = graph.node_map[node2]
    graph.graph[j, i] = 1
    graph.graph[i, j] = 1

    graph.adjust_dpath(i, j)
    
    
    
S = [GraphNode(f"X{i + 1}") for i in range(7)]

for i in range(7):
    S[i].add_attribute("id", i )

S1 = [S[i] for i in range(7) if i not in [2,4]]

S2 = [S[2], S[4]]

G = GeneralGraph(S)

G1 = GeneralGraph(S1)
G2 = GeneralGraph(S2)

G.get_nodes()

G.add_directed_edge(S[0], S[1])
G.add_directed_edge(S[2], S[1])
G.add_directed_edge(S[2], S[3])
G.add_directed_edge(S[4], S[3])
G.add_directed_edge(S[4], S[5])
G.add_directed_edge(S[6], S[5])

G1.add_directed_edge(S[0], S[1])
G1.add_directed_edge(S[6], S[5])
add_bidirected_edge(G1, S[1], S[3])
add_bidirected_edge(G1, S[3], S[5])


GraphUtils.to_pydot(G).write_png("self-comp-G.png")
GraphUtils.to_pydot(G1).write_png("self-comp-G1.png")
GraphUtils.to_pydot(G2).write_png("self-comp-G2.png")

P_G = PAG(G)
P_G1 = PAG(G1)
P_G2 = PAG(G2)


res = SelfCompatibilityScorer._graphical_compatibility(P_G, [P_G1, P_G2])


