from __future__ import annotations

import sys, pathlib
repo_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root / "src" / "external" / "causal-self-compatibility" ))


from dag2pag import dag2pag
from IncrementalGraph import IncrementalGraph
from graphical_metrics import shd_separed

from src.causal_graphs.pag import PAG

from causaldag import DAG
from cdt.data import AcyclicGraphGenerator
from causallearn.graph.GeneralGraph import GeneralGraph

from causallearn.graph.AdjacencyConfusion import AdjacencyConfusion


gen = AcyclicGraphGenerator('linear', npoints=1, nodes=30, dag_type='erdos', expected_degree= 3)
df, digraph = gen.generate()   # X: DataFrame, G: networkx.DiGraph

names = df.columns

names_map = {names[i]: i for i in range(len(names))}

marginal_names = names[10: ]
 
dag = DAG(arcs=set(digraph.edges()))


# 'L' es una variable latente
latent_nodes = set(marginal_names)

obs_names = set(digraph.nodes) - latent_nodes

# Marginalizamos sobre L
mag = dag.marginal_mag(latent_nodes)

pag = dag2pag(digraph, list(obs_names))

bidir_edges = set(tuple(elem) for elem in mag.bidirected)

pag_skel = IncrementalGraph(no_of_var = len(obs_names), initial_graph= GeneralGraph([]), new_node_names= list(obs_names))

for n1, n2 in (mag.directed | bidir_edges):
    pag_skel.add_edge_with_circles(names_map[n1], names_map[n2])


print(shd_separed(pag_skel.G, PAG(pag)))

linkConfusion = AdjacencyConfusion(pag,  pag_skel.G)

print(f"FP: {linkConfusion.get_adj_fp()}. FN: {linkConfusion.get_adj_fn()}.")


