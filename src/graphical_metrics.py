from __future__ import annotations

import sys, pathlib
repo_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root / "src" / "external" / "causal-self-compatibility" ))




from typing import Tuple, List, Iterable, Any, Union, Dict, Set

import networkx as nx


from src.causal_graphs.pag import PAG
from src.causal_graphs.sc_causal_graph import SCCausalGraph

from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.AdjacencyConfusion import AdjacencyConfusion



from EndpointConfusion import EndpointConfusion


ALGORITHMS = [ "CSBS", "PRCDSF", "S-CDFSF", "FCI-FS", "FCI-STABLE"]
GRAPHICAL_SCORE_TYPES = ["l", "a", "c", "t"]
METRICS_NT = ["numCI", "numEdges", "avgSepSize", "execTime",  "HD", "ED", "SHD"]
METRICS_T = ["TP", "FP", "TN", "FN", "PREC", "RECALL", "F1"]
METRICS_SC = ["SC_HD", "SC_HDn", "SC_ED", "SC_EDn", "SC_SHD", "SC_SHDn"]

# Code from: https://github.com/amazon-science/causal-self-compatibility

def shd_separed(ground_truth: Union[nx.DiGraph, SCCausalGraph], estim_pag: PAG) -> Tuple[int, int]:
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

def shd_marginal(marginalised_joint_graph: SCCausalGraph, marginal_graph: SCCausalGraph) -> Tuple[int, int]:
    
    return shd_separed(marginalised_joint_graph, marginal_graph)



"""
ALGORITHMS = [ "CSBS", "PRCDSF", "SCDFSF", "FCIFS"]

METRICS_NT = ["numCI", "numEdges", "avgSepSize", "execTime",  "HD", "ED", "SHD"]
METRICS_T = ["TP", "FP", "TN", "FN", "PREC", "RECALL", "F1"]
METRICS_SC = ["SCHD", "SCHDn", "SCED", "SCEDn", "SCSHD", "SCSHDn, SCF1, SCP, SCR"]
"""
# 0: graph, 1: num_CI_tests, 2: avg_sepset_size, 3: total_exec_time, 4: edges, 5: sep_sets


def get_metrics_nt(ground_truth: Union[nx.DiGraph, SCCausalGraph, GeneralGraph], est_pag: PAG, alg_output: Tuple[Any]) :
    
   
    
    res = []
    
    res.append(alg_output[1]) #numCI
    res.append(len(est_pag.graph.get_graph_edges())) # numEdges
    res.append(alg_output[2]) # avgSepSize
    res.append(alg_output[3]) # execTime
    
    
    
    
    adj_errors, endpoint_errors = shd_marginal(ground_truth, est_pag)
    shd_endpoint = adj_errors + endpoint_errors
    
    res.append(adj_errors) # HD
    res.append(endpoint_errors) # ED
    res.append(shd_endpoint) # SHD
    
    return res

def get_metrics_t(ground_truth: Union[nx.DiGraph, SCCausalGraph], est_graph: PAG) :
    
    
    
    
    endpoint_types = [Endpoint.ARROW, Endpoint.CIRCLE, Endpoint.TAIL]
    
    
    
    
    # METRICS_T = ["TP", "FP", "TN", "FN", "PREC", "RECALL", "F1"]
    
    linkConfusion = AdjacencyConfusion(ground_truth.graph,  est_graph.graph)
    
    link_prec = linkConfusion.get_adj_precision()
    link_recall = linkConfusion.get_adj_recall()
    link_f1 = (link_prec * link_recall)/(link_prec + link_recall)
    
    
    res = [linkConfusion.get_adj_tp(), linkConfusion.get_adj_fp(), linkConfusion.get_adj_tn(), 
           linkConfusion.get_adj_fn(), link_prec, link_recall, link_f1]
    
    for endpoint_t in endpoint_types:
        endpointConfusion = EndpointConfusion(ground_truth.graph, est_graph.graph, endpoint_t)
        
        to_add = [endpointConfusion.get__tp(), endpointConfusion.get__fp(), endpointConfusion.get__tn(), 
                  endpointConfusion.get__fn(), endpointConfusion.get__precision(), endpointConfusion.get__recall(),
                  endpointConfusion.get__F1()]
        
        res.extend(to_add)
        
    return res


def get_alg_marginal_info(ground_truth: Union[nx.DiGraph, SCCausalGraph], est_graph: Union[SCCausalGraph, GeneralGraph], 
                          alg_output: Tuple[Any]):
    
    
    if type(est_graph) == GeneralGraph:
        est_pag = PAG(est_graph)
    elif type(est_graph) != PAG:
        raise NotImplementedError()
    
    if type(ground_truth) == nx.DiGraph:
        ground_truth = est_pag._dag_to_pag(ground_truth)
       
    elif type(ground_truth) == PAG:
        ground_truth = ground_truth
    elif type(ground_truth) != GeneralGraph:
        raise NotImplementedError()

        
    res = get_metrics_nt(ground_truth, est_pag, alg_output)
    
    res.extend(get_metrics_t(ground_truth, est_pag))
    
    return res
    
def get_self_comp_info(num_iter: int, num_var: int, info: List[Any]):
    
    accum_metrics = [0 for i in range(len(METRICS_SC))]
    
    
    num_st = len(GRAPHICAL_SCORE_TYPES)
    num_mt = len(METRICS_T)
    
    
    
    for i in range(num_iter):
        
        accum_metrics[0] +=  info[i*num_mt*num_st + 4] # accumulate SC_HD
        accum_metrics[1] +=  info[i*num_mt*num_st + 4]/num_var # accumulate SC_HD divided by num vars
        accum_metrics[2] +=  info[i*num_mt*num_st + 5] # accumulate SC_ED
        accum_metrics[3] +=  info[i*num_mt*num_st + 5]/num_var # accumulate SC_ED divided by num vars
        accum_metrics[4] +=  info[i*num_mt*num_st + 6] # accumulate SC_SHD
        accum_metrics[5] +=  info[i*num_mt*num_st + 6]/num_var # accumulate SC_SHD divided by num vars
        
    return [metric/num_iter for metric in accum_metrics] # take the average
        
        
        
            
            
    
    
    
    """
    GRAPHICAL_SCORE_TYPES = ["l", "a", "c", "t"]
    METRICS_NT = [0: "numCI", 1: "numEdges", 2: "avgSepSize", 3: "execTime",  4: "HD", 5: "ED", 6: "SHD"]
    METRICS_T = [(6 + 7n + 1): "TP", (6 + 7n + 1) "FP", "TN", "FN", "PREC", "RECALL", "F1"]
    METRICS_SC = [0: "SC_HD", 1:"SC_HDn", 2: "SC_ED", 3:"SC_EDn", 4: "SC_SHD", 5:"SC_SHDn"]
    """
    
    
    
    
    
    