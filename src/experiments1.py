#!/home/chema/anaconda3/envs/FCI-FS_env/bin/python

"""
experimen

# -*- coding: utf-8 -*-

Created on Sun Jan 26 19:26:47 2025

@author: chdem
"""




import csv
import numpy as np
import time
import gc

import warnings
warnings.filterwarnings("error", message="The truth value of an empty array is ambiguous*")

from causallearn.utils.cit           import fisherz      # ∼O(n³) linear-Gaussian CI test
from causallearn.graph.GeneralGraph import GeneralGraph 
from cdt.data import AcyclicGraphGenerator

from CSBS_FS import csbs_fs
from PRCDSF_FS import prcdsf_fs
from S_CDFSF_FS import s_cdfsf_fs
from CSSU_FS import cssu_fs
from FCI_FS import fci_fs
import graphical_metrics as g_m
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


"""

#!/usr/bin/env python3
import argparse

# Crear parser
parser = argparse.ArgumentParser(description="Ejecuta experimento con parámetros configurables.")

parser.add_argument("--numPVal", type=int, required=True, help="Número de pVal")
parser.add_argument("--numInstances", type=int, required=True, help="Número de instancias de datos")
parser.add_argument("--numVars", type=int, required=True, help="Número de variables")


args = parser.parse_args()
# Asignar a variables
numPVal = args.numPVal
NUM_INSTANCES = args.numInstances
NUM_VARS = args.numVars

"""




numPVal = 0
NUM_INSTANCES = 300
NUM_VARS = 10


INITIAL_P = 0.1

NEIGHBORHOOD_SIZE = 2 
MAX_ITER = 1e3
CI_TEST = fisherz


NUM_DATASET_SIZES = 1
NUM_RANDOM_DAGS = 10
NUM_ORDERS = 1


NUM_PERCENTAGE = 5
PERCENT_STEP = 0.20
percentList = [round((k+1)*PERCENT_STEP, 2) for k in range(NUM_PERCENTAGE)]

ALPHA = INITIAL_P/(2**numPVal)

time0 = time.time()


time_marginal = 0

if NUM_VARS % NUM_PERCENTAGE != 0:
   raise AssertionError("NUM_VARS should be divisible by NUM_PERCENTAGE")

# Empty the file
file = open(f"../../logs/log_pVal{numPVal}_dataSize{NUM_INSTANCES}_nVars{NUM_VARS}_nbSize{NEIGHBORHOOD_SIZE}.txt", mode='w')
file.close()


with open(f"../../logs/log_pVal{numPVal}_dataSize{NUM_INSTANCES}_nVars{NUM_VARS}_nbSize{NEIGHBORHOOD_SIZE}.txt", "a", buffering=1) as file1:  # line-buffered in text mode
   
        
    for tam_mult in range(0, NUM_DATASET_SIZES): 
        
        dataset_size = NUM_INSTANCES*(2**tam_mult)
        
        with open(f"output_pVal{numPVal}_dataSize{NUM_INSTANCES}_nVars{NUM_VARS}_nbSize{NEIGHBORHOOD_SIZE}.csv", mode='w', newline='') as file2:
            writer = csv.writer(file2)
            #Write a header row
            
            column_names = []
        
            for alg_name in g_m.ALGORITHMS:
                for percentage in percentList:
                    suffix = "_" + alg_name + "_" + str(int(percentage*100))
                    
                    for metric_nt in g_m.METRICS_NT:
                        column_name = metric_nt + suffix
                        column_names.append(column_name)
                        
                    for metric_type in g_m.GRAPHICAL_SCORE_TYPES:
                        suffix2 = suffix + "_" + metric_type
                        
                        for metric_t in g_m.METRICS_T:
                            column_name = metric_t + suffix2
                            column_names.append(column_name)
                            
                suffix = "_" + alg_name 
                for metric_sc in g_m.METRICS_SC:
                    column_name = metric_sc + suffix
                    column_names.append(column_name)
            sumar = 0
            writer.writerow(column_names)
            
            for j in range(0, NUM_RANDOM_DAGS):
                # We halve NEIGHBORHOOD_SIZE because cdt doubles the expected neightborhood size () (We are wainting untill merge pull request #169 is accepted to solve this BUG #168)
                gen = AcyclicGraphGenerator('linear', npoints=NUM_INSTANCES, nodes=NUM_VARS, dag_type='erdos', expected_degree= NEIGHBORHOOD_SIZE/2) 
                
                df, digraph = gen.generate()   # X: DataFrame, G: networkx.DiGraph
                            
                names = df.columns
             
                for i in range(0, NUM_ORDERS):
        
                        random_permutation = np.random.permutation(NUM_VARS)
                        df_permuted = df.iloc[:, random_permutation]
                        
                        permuted_names = df_permuted.columns
                        data = df_permuted.to_numpy(copy=False)
                        
                        del df_permuted
                        
                        output_csbs = (GeneralGraph([]), 0, 0, 0)
                        output_prcdsf = (GeneralGraph([]), 0, 0, 0)
                        output_scdfsf = (GeneralGraph([]), 0, 0, 0)
                        output_cssu = (GeneralGraph([]), 0, 0, 0)
                        output_fci_fs = (GeneralGraph([]), 0, 0, 0, [], {})
                        output_fci_stable = (GeneralGraph([]), 0, 0, 0, [], {})
                        
                        csbs_info = []
                        prcdsf_info = []
                        scdfsf_info = []
                        cssu_info = []
                        fci_fs_info = []
                        fci_stable_info = []
                        
                        for percentage in percentList :
                            
                            
                            start_pos =int(round((percentage - PERCENT_STEP)*NUM_VARS))
                            
                            
                            
                            end_pos = int(round(percentage*NUM_VARS))
                            data_marginal = data[:, 0:end_pos]
                            names_marginal = permuted_names[0: end_pos]
                            names_latent = permuted_names[end_pos : ]
                            new_names = permuted_names[ start_pos:end_pos]
                            
                            
                            
                            time_marginal0 = time.time()
                            ground_truth = dag2pag(digraph, names_marginal)
                            time_marginal += time.time() - time_marginal0
                            
                      
                            
                            output_csbs = csbs_fs(data_marginal, independence_test_method=CI_TEST, alpha= ALPHA, initial_graph= output_csbs[0], new_node_names= new_names ,verbose = False, max_iter = MAX_ITER) 
                            output_prcdsf = prcdsf_fs(data_marginal, independence_test_method=CI_TEST, alpha= ALPHA,   initial_graph= output_prcdsf[0], new_node_names= new_names ,verbose = False, max_iter = MAX_ITER)
                            output_scdfsf = s_cdfsf_fs(data_marginal, independence_test_method=CI_TEST, alpha= ALPHA,   initial_graph= output_scdfsf[0], new_node_names= new_names ,verbose = False, max_iter = MAX_ITER)                    
                            output_cssu = cssu_fs(data_marginal, alpha= ALPHA,  initial_graph= output_cssu[0], new_node_names= new_names, verbose=False, max_iter= MAX_ITER)
                            output_fci_fs = fci_fs(data_marginal, independence_test_method=CI_TEST,  initial_sep_sets = output_fci_fs[5], alpha= ALPHA,  initial_graph= output_fci_fs[0] ,  new_node_names= new_names ,verbose = False)
                            output_fci_stable = fci_fs(data_marginal, independence_test_method=CI_TEST, initial_sep_sets = {}, alpha= ALPHA,  initial_graph = GeneralGraph([]), new_node_names = names_marginal, verbose = False)
                            
                            file1.write(f"Percentage: {percentage}. ExecTimes:  csbs - {output_csbs[3]}. prcdsf - {output_prcdsf[3]}. scdfsf - {output_scdfsf[3]}. cssu - {output_cssu[3]}. FCI-FS - {output_fci_fs[3]}. FCI - {output_fci_stable[3]}\n")
                            
                            #Metrics of the marginal models
                            csbs_info.extend(g_m.get_alg_marginal_info(ground_truth, output_csbs[0], output_csbs))
                            prcdsf_info.extend(g_m.get_alg_marginal_info(ground_truth, output_prcdsf[0], output_prcdsf))
                            scdfsf_info.extend(g_m.get_alg_marginal_info(ground_truth, output_scdfsf[0], output_scdfsf))
                            cssu_info.extend(g_m.get_alg_marginal_info(ground_truth, output_cssu[0], output_cssu))
                            fci_fs_info.extend(g_m.get_alg_marginal_info(ground_truth, output_fci_fs[0], output_fci_fs))
                            fci_stable_info.extend(g_m.get_alg_marginal_info(ground_truth, output_fci_stable[0], output_fci_stable))
                            
                          
                            
                        csbs_info.extend(g_m.get_self_comp_info(len(percentList), NUM_VARS, csbs_info))
                        prcdsf_info.extend(g_m.get_self_comp_info(len(percentList), NUM_VARS, prcdsf_info))
                        scdfsf_info.extend(g_m.get_self_comp_info(len(percentList), NUM_VARS, scdfsf_info))
                        cssu_info.extend(g_m.get_self_comp_info(len(percentList), NUM_VARS, cssu_info))
                        fci_fs_info.extend(g_m.get_self_comp_info(len(percentList), NUM_VARS, fci_fs_info))
                        fci_stable_info.extend(g_m.get_self_comp_info(len(percentList), NUM_VARS, fci_stable_info))
                        
                       
                        new_row = csbs_info + prcdsf_info + scdfsf_info + cssu_info + fci_fs_info + fci_stable_info
                        writer.writerow(new_row)
                        
                        del csbs_info, prcdsf_info, scdfsf_info, cssu_info, fci_fs_info, fci_stable_info, new_row, output_csbs, output_prcdsf, output_scdfsf, output_cssu, output_fci_fs, output_fci_stable
                        
                        gc.collect()
                        
                        file2.flush()
                        os.fdatasync(file2.fileno())
              
                        file1.write(f"DATA: {dataset_size}. DAG {j+1} of  {NUM_RANDOM_DAGS}. ORDER {i+1} of {NUM_ORDERS}. \n")
                        file1.write(f"DATA: {dataset_size}. AvgTime: {(time.time() - time0)/(j+1) :.3f} seconds. AvgTimeMarginal {time_marginal/(j+1):.3f} seconds. \n")
                        file1.flush()              # flush Python’s buffer
                        os.fsync(file1.fileno())   # force OS to write to disk
                        
                       
                        
                    
                del  df
                gc.collect()
                
                        
                    
    
    file1.write(f"FINISHED. AvgTime: {(time.time() - time0)/NUM_RANDOM_DAGS :.3f} seconds. AvgTimeMarginal {time_marginal/NUM_RANDOM_DAGS:.3f} seconds. \n")
    file1.flush()              # flush Python’s buffer
    os.fsync(file1.fileno())   # force OS to write to disk
