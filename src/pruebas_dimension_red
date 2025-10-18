#!/home/chema/anaconda3/envs/FCI-FS_env/bin/python

"""
experiments1
"""



# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 19:26:47 2025

@author: chdem
"""




import csv
import numpy as np
import time
from datetime import datetime
import networkx as nx
import sys

from causallearn.utils.cit           import fisherz      # ∼O(n³) linear-Gaussian CI test
from causallearn.utils.GraphUtils    import GraphUtils
from causallearn.graph.GeneralGraph import GeneralGraph 
from cdt.data import AcyclicGraphGenerator
from causallearn.utils.cit import CIT, d_separation

from CSBS_FS import csbs_fs
from PRCDSF_FS import prcdsf_fs
from S_CDFSF_FS import s_cdfsf_fs
from CSSU_FS import cssu_fs
from FCI_FS import fci_fs
import graphical_metrics as g_m
import auxiliary as aux
from to_PAG import to_PAG
from dag2pag import dag2pag

# --- FORZAR 1 HILO EN TODAS LAS LIBRERÍAS NUMÉRICAS ---
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"
os.environ["TBB_NUM_THREADS"] = "1"

# (opcional) evita que MKL/OMP se reconfiguren tras fork
os.environ["MKL_DYNAMIC"] = "FALSE"
os.environ["OMP_DYNAMIC"] = "FALSE"

# Limitar backends subyacentes (Sklearn/NumPy/Torch) a 1 hilo
try:
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
except Exception:
    pass

# Si Torch está instalado (CDT a veces lo usa)
try:
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass
# --- FIN BLOQUE ---



import time 

numPVal =  1


INITIAL_P = 0.1
NUM_VARS = 200

NEIGHBORHOOD_SIZE = 1 # We halve NEIGHBORHOOD_SIZE because cdt doubles the expected neightborhood size
MAX_ITER = 1e3
CI_TEST = fisherz


NUM_DATASET_SIZES = 1
NUM_INSTANCES = 1
NUM_RANDOM_DAGS = 10
NUM_ORDERS = 1

NUM_PERCENTAGE = 5
PERCENT_STEP = 0.20
percentList = [round((k+1)*PERCENT_STEP, 2) for k in range(NUM_PERCENTAGE)]

ALPHA = INITIAL_P/(2**numPVal)

time0 = time.time()


time_marginal = 0


for j in range(0, NUM_RANDOM_DAGS):
    
    print(f"NumVars: {NUM_VARS} DAG {j} of {NUM_RANDOM_DAGS} \n \n", file=sys.stderr)
        
    gen = AcyclicGraphGenerator('linear', npoints=NUM_INSTANCES, nodes=NUM_VARS, dag_type='erdos', expected_degree= NEIGHBORHOOD_SIZE)
    df, digraph = gen.generate()   # X: DataFrame, G: networkx.DiGraph
                        
    names = df.columns
            
    full_ground_truth = to_PAG(digraph, names)
    random_permutation = np.random.permutation(NUM_VARS)
    df_permuted = df.iloc[:, random_permutation]
   
    permuted_names = df_permuted.columns
    data = df_permuted.values
    for percentage in percentList :
        print(f"Percentage: {percentage}")
        start_pos =int((percentage - PERCENT_STEP)*NUM_VARS)
        end_pos = int(percentage*NUM_VARS)
        data_marginal = data[:, 0:end_pos]
        names_marginal = permuted_names[0: end_pos]
        names_latent = permuted_names[end_pos : ]
        new_names = permuted_names[ start_pos:end_pos]
        print("Everything ok")
                        
        time_marginal0 = time.time()
                        
        ground_truth = dag2pag(digraph, names_marginal)
                        
        time_marginal += time.time() - time_marginal0
                        
        print(f"Time Marginal: {time_marginal}.", file=sys.stderr)
       
                
print(f"AvgTime: {(time.time() - time0)/NUM_RANDOM_DAGS :.3f} seconds. AvgTimeMarginal {time_marginal/NUM_RANDOM_DAGS:.3f} seconds.")
