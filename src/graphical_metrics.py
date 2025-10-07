from __future__ import annotations

import sys, pathlib
repo_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root / "src" / "external" / "causal-self-compatibility" ))




from typing import Tuple, List, Iterable, Any, Union, Dict, Set

import networkx as nx


from causal_graphs.pag import PAG

from causallearn.graph.Endpoint import Endpoint


from src.causal_graphs.sc_causal_graph import SCCausalGraph



def shd_separed(ground_truth: Union[nx.DiGraph, SCCausalGraph], estim_pag: PAG) -> int:
    if type(ground_truth) == nx.DiGraph:
        ground_truth = estim_pag._dag_to_pag(ground_truth)
    elif type(ground_truth) == PAG:
        ground_truth = ground_truth.graph
    else:
        raise NotImplementedError()
    adj_errors = 0
    endpoint_errors = 0
    for x, y in [(x.get_name(), y.get_name()) for i, x in enumerate(ground_truth.get_nodes()) for j, y in
                 enumerate(ground_truth.get_nodes()) if i < j]:
        
        if ground_truth.is_adjacent_to(ground_truth.get_node(x), ground_truth.get_node(y)):
            gt_edge = ground_truth.get_edge(ground_truth.get_node(x), ground_truth.get_node(y))
            if not estim_pag.graph.is_adjacent_to(estim_pag.graph.get_node(x), estim_pag.graph.get_node(y)):
                adj_errors += 1
                endpoint_errors += 1 if gt_edge.get_endpoint1() != Endpoint.CIRCLE else 0
                endpoint_errors += 1 if gt_edge.get_endpoint2() != Endpoint.CIRCLE else 0
            else:
                hat_edge = estim_pag.graph.get_edge(estim_pag.graph.get_node(x), estim_pag.graph.get_node(y))
                if gt_edge.get_endpoint1() != hat_edge.get_endpoint1():
                    endpoint_errors += 1
                if gt_edge.get_endpoint2() != hat_edge.get_endpoint2():
                    endpoint_errors += 1
        else:
            if estim_pag.graph.is_adjacent_to(estim_pag.graph.get_node(x), estim_pag.graph.get_node(y)):
                hat_edge = estim_pag.graph.get_edge(estim_pag.graph.get_node(x), estim_pag.graph.get_node(y))
                adj_errors += 1
                endpoint_errors += 1 if hat_edge.get_endpoint1() != Endpoint.CIRCLE else 0
                endpoint_errors += 1 if hat_edge.get_endpoint2() != Endpoint.CIRCLE else 0
    return adj_errors, endpoint_errors

def shd_marginal(joint_graph: SCCausalGraph, marginal_graph: SCCausalGraph) -> float:
    marginalised_joint_graph = joint_graph.marginalize(marginal_graph.variables())
    
    return shd_separed(marginalised_joint_graph, marginal_graph)