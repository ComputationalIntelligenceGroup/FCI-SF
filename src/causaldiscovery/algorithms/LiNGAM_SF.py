# -*- coding: utf-8 -*-
"""
Created on Sun Feb  8 16:19:49 2026

@author: chdem
"""

from __future__ import annotations

import warnings
from typing import Dict, FrozenSet, Tuple, List, Union, Iterable, Set, Optional
from numpy import ndarray
import numpy as np
import time
import pandas as pd

from causallearn.graph.Graph import Graph
from causallearn.graph.GeneralGraph import GeneralGraph 

from causallearn.utils.cit import *

import sys, pathlib
repo_root = pathlib.Path(__file__).resolve().parents[3]
sys.path.append(str(repo_root / "src"))

from causaldiscovery.CItest.noCache_CI_Test import myTest
from causaldiscovery.graphs.IncrementalGraph import IncrementalGraph
from causaldiscovery.utils.orientation import orientation
from causaldiscovery.algorithms.MBCS import mbcs_skeleton
from causaldiscovery.algorithms.CSBS import csbs
from causaldiscovery.CItest.noCache_CI_Test import myTest
import networkx as nx

Var = Union[int, str]
Pair = Tuple[Var, Var]

# mb.G, num_CI_tests, avg_sepset_size, total_exec_time


def lingam_sf(data: ndarray, independence_test_method: CIT_Base, alpha1: float = 0.05,  alpha2: float = 0.001,  r = 0.7,
            initial_graph: GeneralGraph = None, old_latents_pairs=None,  verbose: bool = False,  new_node_names:List[str] = None, **kwargs) -> Tuple[Graph, int, int]:
    
    nCI = 0 
    
    if initial_graph is None:
        initial_graph = GeneralGraph([])
        
    Gt1 = IncrementalGraph(0, initial_graph)
    
    if Gt1.G.get_num_nodes() == 0:
        Gt1, num_CI_tests, sepset_size = mbcs_skeleton(data = data, independence_test_method = independence_test_method, alpha = alpha1, 
                           verbose = verbose, new_node_names = new_node_names)
        nCI += num_CI_tests
    else:
        Gt1, num_CI_tests, sepset_size = csbs(data, independence_test_method=independence_test_method, alpha= alpha1, 
                           initial_graph= initial_graph, new_node_names= new_node_names ,verbose = verbose, 
                           max_iter = 0, only_skel = True) 
        nCI += num_CI_tests
        
    
    print("Skeleton done!")
    
    
    Gt2, Lalter_nodes, T_pairs, num_CI_tests = icdplv(data, Gt1, r, alpha2, verbose=verbose)
    
    nCI += num_CI_tests

    Lt_pairs, LC_map, num_CI_tests = dlc_pairwise(
        G=Gt2,
        X=data,
        Lalter_pairs=T_pairs,         
        sig=alpha2,
        )
    
    nCI += num_CI_tests
        
    return Gt2, Lt_pairs, nCI

def compute_residual(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Residual r_x(y) = x - (cov(x,y)/var(y)) * y
    x, y: shape (n_samples,)
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    # center
    xc = x - x.mean()
    yc = y - y.mean()

    var_y = np.mean(yc * yc) + eps
    cov_xy = np.mean(xc * yc)
    beta = cov_xy / var_y

    return x - beta * y

def judge_direction(row1: np.ndarray, row2: np.ndarray) -> int:
    """
    row1, row2: 1D binary arrays (support rows) with same length
    Returns:
      +1 means var1 -> var2
      -1 means var2 -> var1
       0 undecided
    """
    row1 = np.asarray(row1).astype(int).reshape(-1)
    row2 = np.asarray(row2).astype(int).reshape(-1)
    assert row1.shape == row2.shape

    n0_star = int(np.sum((row1 == 0) & (row2 == 1)))
    nstar_0 = int(np.sum((row1 == 1) & (row2 == 0)))

    if (n0_star > 0) and (nstar_0 == 0):
        return +1
    if (nstar_0 > 0) and (n0_star == 0):
        return -1
    return 0




def icdplv(
    data: ndarray,
    incremental_graph,
    r = 0.7,
    sig = 0.001,
    verbose = False,
    orientation_kwargs=None,
):
    """
    Parameters
    ----------
    incremental_graph : your object
        Must provide:
    X : ndarray (n_samples, p)
    r : float sampling ratio passed to orientation()
    sig : float significance threshold for Proposition 1
    orientation_kwargs : dict forwarded to orientation()

    Returns
    -------
    Gt : nx.DiGraph
        Directed edges accepted by Prop1 and Prop2
    Lalter : set
        Endpoints of unresolved edges after Prop1 (candidate latent confounders)
    num_CI_tests: int
        Number of performed CI test
    """
    if orientation_kwargs is None:
        orientation_kwargs = {}
        
    X = np.asarray(data, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError("data must be 2D array (n_samples, p)")
    n_samples, p = X.shape
    
 

    if not (0 < r <= 1):
        raise ValueError("r must be in (0,1]")
        
    num_CI_tests = 0

    node_names = incremental_graph.G.get_node_names()
    Gt = nx.DiGraph()
    Gt.add_nodes_from(node_names)

    T = []          # unresolved edges after Prop1
    Lalter = set()  # endpoints of unresolved edges after Prop1

    # -------------------------
    # Phase A: Proposition 1 
    # -------------------------
    U = incremental_graph.get_numerical_edges()

    for (a, b) in U:
        xa = X[:, a]
        xb = X[:, b]
        
        

        ra_b = compute_residual(xa, xb)  
        rb_a = compute_residual(xb, xa)  

        # one DF per edge, then one CI tester
        df_edge = pd.DataFrame({
            "xa": xa,
            "xb": xb,
            "ra_b": ra_b,
            "rb_a": rb_a,
        })
        ci = myTest(df_edge)

        idx_xa, idx_xb, idx_ra_b, idx_rb_a = 0, 1, 2, 3

        p1 = ci(idx_ra_b, idx_xb, set())  # ra_b ⟂ xb ?
        p2 = ci(idx_rb_a, idx_xa, set())  # rb_a ⟂ xa ?
        
        num_CI_tests += 2

        na = node_names[a]
        nb = node_names[b]

        if (p1 > sig) and (p2 <= sig):
            print("b -> a")
            # b -> a
            Gt.add_edge(nb, na)
        elif (p2 > sig) and (p1 <= sig):
            print("a -> b")
            # a -> b
            Gt.add_edge(na, nb)
        else:
            print("Common confounder (append T)")
            T.append((a, b))
            Lalter.add(a)
            Lalter.add(b)

    # -------------------------
    # Phase B: Proposition 2 
    # -------------------------
    def add_edge_if_no_cycle(u, v):
        # adding u->v creates cycle iff v can already reach u
        if nx.has_path(Gt, v, u):
            return False
        Gt.add_edge(u, v)
        return True
    
    counter = 0
    for (a, b) in T:
        counter += 1
        print(f"Edge {counter} of {len(U)}")
        data_ab = X[:, [a, b]]  # (n_samples, 2)
        
        if verbose:
            print("Going for orientation")
        
        support, _ = orientation(data_ab, r,verbose = verbose, **orientation_kwargs)  # support is 2x2

        ori = judge_direction(support[0, :], support[1, :])

        na = node_names[a]
        nb = node_names[b]

        if ori == 1:
            print("Phase 2: ori == 1")
            # a -> b
            add_edge_if_no_cycle(na, nb)
        elif ori == -1:
            print("Phase 2: ori == -1")
            # b -> a
            add_edge_if_no_cycle(nb, na)
        else:
            print("Phase 2: ori == 0 (else)")
            # undecided: do nothing (equivalent to "delete undirected edge" for output graph)
            pass

    return Gt, Lalter, T, num_CI_tests



# Reusa tu compute_residual y myTest
# from causaldiscovery.CItest.noCache_CI_Test import myTest




def _as_pairs(pairs: Iterable[Pair]) -> Set[Pair]:
    """Normaliza pares: (a,b) con a!=b, y ordena para evitar duplicados (a,b) vs (b,a)."""
    out: Set[Pair] = set()
    for a, b in pairs:
        if a == b:
            continue
        # orden canónico para tratar el confounder latente como no-direccional
        out.add((a, b) if str(a) < str(b) else (b, a))
    return out

def _common_parents(G: nx.DiGraph, a: Var, b: Var) -> Set[Var]:
    """Padres comunes observados: pa -> a y pa -> b."""
    if a not in G or b not in G:
        return set()
    return set(G.predecessors(a)).intersection(set(G.predecessors(b)))


def _prop1_says_no_latent(x: np.ndarray, y: np.ndarray, sig: float) -> bool:
    """
    Implementa el chequeo tipo Proposition 1 usado en tu icdplv:
      ra_y = res(x|y), rb_x = res(y|x)
      Si exactamente una independencia se cumple (p>sig) y la otra no (p<=sig),
      interpretamos relación causal (estructura 1) => NO latent confounder.
    """
    ra_y = compute_residual(x, y)
    rb_x = compute_residual(y, x)

    df = pd.DataFrame({"x": x, "y": y, "ra_y": ra_y, "rb_x": rb_x})
    ci = myTest(df)

    # índices columnas: x=0,y=1,ra_y=2,rb_x=3
    p1 = ci(2, 1, set())  # ra_y ⟂ y ?
    p2 = ci(3, 0, set())  # rb_x ⟂ x ?

    # "estructura 1" (orientable) => no confounder latente necesario
    if (p1 > sig and p2 <= sig) or (p2 > sig and p1 <= sig):
        return True
    return False


def dlc_pairwise(
    G: nx.DiGraph,
    X: np.ndarray,
    Lalter_pairs: Iterable[Pair],
    sig: float = 0.001,
    old_latents_pairs: Optional[Iterable[Pair]] = None,
    copy_data: bool = True,
) -> Tuple[Set[Pair], Dict[Var, Set[Var]]]:
    """
    DLC manteniendo SIEMPRE latentes por pares (sin agrupar por Property 2).

    Parameters
    ----------
    G : nx.DiGraph
        Grafo causal actual (dirigido).
    X : ndarray (n_samples, p)
        Datos observados. Si usas índices, deben corresponder a columnas de X.
    Lalter_pairs : iterable de (vi, vj)
        Candidatos a confounder latente por pares.
        (Recomendado: usa T de icdplv, que ya son pares.)
    sig : float
        Umbral de significación para el chequeo de Prop. 1.
    old_latents_pairs : iterable de (vi, vj)
        Latentes anteriores (L_{t-1}) en formato por pares.
    copy_data : bool
        Si True, no modifica X original; trabaja sobre copia interna.

    Returns
    -------
    Lt_pairs : set de (vi, vj)
        Pares finales retenidos como confounders latentes.
    LC_map : dict var -> set(confusores observados)
        Mapa de confusores observados usados para residualizar.
    num_CI_test: int
        Number of CI tests.
    """
    if X.ndim != 2:
        raise ValueError("X must be 2D (n_samples, p)")
        
    num_CI_test = 0

    Xwork = X.copy() if copy_data else X

    # 1) Inicialización + unión con L_{t-1}
    cand = _as_pairs(Lalter_pairs)
    if old_latents_pairs is not None:
        cand |= _as_pairs(old_latents_pairs)

    # 2) Construir LC(var): confusores observados (padres comunes) por cada extremo
    LC_map: Dict[Var, Set[Var]] = {}
    for a, b in cand:
        parents = _common_parents(G, a, b)
        if parents:
            LC_map.setdefault(a, set()).update(parents)
            LC_map.setdefault(b, set()).update(parents)

    # 3) Residualizar cada variable t por sus confusores observados LC(t)
    #    (igual que Matlab: sucesivamente para cada parent)
    for t, parents in LC_map.items():
        # si t es nombre, no puedes indexar X: en ese caso, necesitas un mapping nombre->col
        if not isinstance(t, (int, np.integer)):
            raise TypeError(
                "dlc_pairwise: si usas nombres de nodos (str), "
                "necesitas convertirlos a índices de columna antes de llamar."
            )
        xt = Xwork[:, int(t)]
        for pa in parents:
            if not isinstance(pa, (int, np.integer)):
                raise TypeError(
                    "dlc_pairwise: confusores observados deben ser índices si X es matriz."
                )
            xt = compute_residual(xt, Xwork[:, int(pa)])
        Xwork[:, int(t)] = xt

    # 4) Re-chequear cada par tras eliminar confusores observados.
    #    Si Prop1 indica estructura 1 (orientable), eliminamos el candidato.
    kept: Set[Pair] = set()
    for a, b in cand:
        if not (isinstance(a, (int, np.integer)) and isinstance(b, (int, np.integer))):
            raise TypeError(
                "dlc_pairwise: pares deben ser índices de columna si X es matriz."
            )
        xa = Xwork[:, int(a)]
        xb = Xwork[:, int(b)]
        
        

        no_latent = _prop1_says_no_latent(xa, xb, sig=sig)
        num_CI_test += 2
        if not no_latent:
            kept.add((a, b) if str(a) < str(b) else (b, a))

    return kept, LC_map, num_CI_test

