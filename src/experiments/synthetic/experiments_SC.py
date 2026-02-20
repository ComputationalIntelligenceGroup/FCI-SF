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

import sys, pathlib
repo_root = pathlib.Path(__file__).resolve().parents[3]
sys.path.append(str(repo_root / "src"))
print(repo_root)
sys.path.append(str(repo_root / "src" / "external" / "causal-self-compatibility/src" ))


from causaldiscovery.algorithms.FCI_SF import fci_sf


import causaldiscovery.metrics.graphical_metrics as g_m
from causaldiscovery.graphs.dag2pag import dag2pag
from causaldiscovery.CItest.noCache_CI_Test import myTest
from causaldiscovery.utils.auxiliary import make_bn_truth_and_sample
import os
import argparse







# --- FORZAR 1 HILO EN TODAS LAS LIBRERÍAS NUMÉRICAS ---

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

# Crear parser
parser = argparse.ArgumentParser(description="Ejecuta experimento con parámetros configurables.")

parser.add_argument("--numPVal", type=int, required=True, help="Number of pVal")
parser.add_argument("--numInstances", type=int, required=True, help="Data size")
parser.add_argument("--numVars", type=int, required=True, help="Number of variables")
parser.add_argument("--numDAGs", type=int, required=True, help="Number of random DAGs")


args = parser.parse_args()
# Asignar a variables
numPVal = args.numPVal
NUM_INSTANCES = args.numInstances
NUM_VARS = args.numVars
NUM_RANDOM_DAGS = args.numDAGs
"""

numPVal = 1
NUM_INSTANCES = 1e3
NUM_VARS = 600
NUM_RANDOM_DAGS = 1e3

INITIAL_P = 0.1

NEIGHBORHOOD_SIZE = 2 
MAX_ITER = 1e3



# Do not touch this parameter
NUM_DATASET_SIZES = 1

NUM_PERCENTAGES = [6, 12, 15, 20, 30, 60, 120]

percentages = [[round(1/num_sep, 2) for k in range(num_sep)] for num_sep in NUM_PERCENTAGES]

ALPHA = INITIAL_P/(2**numPVal)

time0 = time.time()


time_marginal = 0

time_iter = 0

# BASE RANDOM GENERATOR TO GUARANTEE REPRODUCIBILITY
base_rng = np.random.default_rng(123)


if any(NUM_VARS % num_percentage for num_percentage in NUM_PERCENTAGES):
   raise AssertionError("NUM_VARS should be divisible by NUM_PERCENTAGE")

# Empty the file
file = open(f"../../../../logs/log_SC__pVal{numPVal}_dataSize{NUM_INSTANCES}_nVars{NUM_VARS}_nbSize{NEIGHBORHOOD_SIZE}_nDAGs{NUM_RANDOM_DAGS}.txt", mode='w')
file.close()


with open(f"../../../../logs/log_SC_pVal{numPVal}_dataSize{NUM_INSTANCES}_nVars{NUM_VARS}_nbSize{NEIGHBORHOOD_SIZE}_nDAGs{NUM_RANDOM_DAGS}.txt", "a", buffering=1) as file1:  # line-buffered in text mode

   
        
    for tam_mult in range(0, NUM_DATASET_SIZES): 
        
        dataset_size = NUM_INSTANCES*(2**tam_mult)
        
        with open(f"output_SC_pVal{numPVal}_dataSize{NUM_INSTANCES}_nVars{NUM_VARS}_nbSize{NEIGHBORHOOD_SIZE}.csv", mode='w', newline='') as file2:
            writer = csv.writer(file2)
            #Write a header row
            
            column_names = []
        
            for alg_name in [ "FCI-Stable"]:

                for percentList in percentages:

                    for i_pl in range(len(percentList)):
                        percentage = percentList[i_pl]
                        suffix = "_" + alg_name + "_" + str(int(percentage*NUM_VARS)) + "_" + str(int(i_pl))
                        
                        # 0:TP, 1:FP, 2:TN, 3:FN,
                        for metric_nt in ["HD", "SHD", "NE", "TP", "FP", "TN", "FN"]:
                            column_name = metric_nt + suffix
                            column_names.append(column_name)
                            
                            
                       
        
            sumar = 0
            writer.writerow(column_names)
            
            for j in range(0, NUM_RANDOM_DAGS):
                # We halve NEIGHBORHOOD_SIZE because cdt doubles the expected neightborhood size () (We are wainting untill merge pull request #169 is accepted to solve this BUG #168)
                
                digraph, df = make_bn_truth_and_sample(
                    base_rng = base_rng ,
                    num_vars = NUM_VARS,
                    expected_degree = NEIGHBORHOOD_SIZE,
                    n_samples = int(dataset_size)
                )                
                            
                names = df.columns
                
                fci_stable_info = []
             
                for i in range(0, len(NUM_PERCENTAGES)):
                    
                        val_percent = NUM_PERCENTAGES[i]
                    
                        time_iter = time.time()
        
                        random_permutation = np.random.permutation(NUM_VARS)
                        df_copied = df.copy()

                        
                        copied_names = df_copied.columns
                        data = df_copied.to_numpy(copy=False)
                        
                        
                        CI_test = myTest(df_copied)
                        
                      
                        output_fci_stable = (GeneralGraph([]), 0, 0, 0, [], {})
                        
                        

                        
                        
                        for i_pl in range(0, val_percent) :
                            
                            percentage = percentList[i_pl]
                            
                           
                            
                            start_pos = int(round(i_pl/val_percent * NUM_VARS))
                            end_pos   = int(round((i_pl+1)/val_percent * NUM_VARS))
                            
                            
                            data_marginal = data[:, start_pos:end_pos]
                            names_marginal = copied_names[start_pos: end_pos]                            
                            
                            time_marginal0 = time.time()
                            ground_truth = dag2pag(digraph, names_marginal)
                            time_marginal += time.time() - time_marginal0
                            
                      

                            
                         
                            # === Instrumentación alrededor de cada llamada ===
                            output_fci_stable = fci_sf(data_marginal, independence_test_method=CI_test, initial_sep_sets = {}, alpha= ALPHA,  initial_graph = GeneralGraph([]), new_node_names = names_marginal, verbose = False)
                         
                            
                            
                            file1.write(f"Percentage: {percentage}. ExecTimes:  FCI - {output_fci_stable[3]}\n")
                            
                            #Metrics of the marginal models
                            
                            fci_stable_res_nt = g_m.get_metrics_nt(ground_truth, output_fci_stable[0], output_fci_stable)
                            fci_stable_res_t =g_m.get_metrics_t(ground_truth,  output_fci_stable[0])
                           
                            fci_stable_HD_SHD = [fci_stable_res_nt[4], fci_stable_res_nt[6], fci_stable_res_nt[1],fci_stable_res_t[0], fci_stable_res_t[1], fci_stable_res_t[2], fci_stable_res_t[3] ]

                            fci_stable_info.extend(fci_stable_HD_SHD)
                            
                        
                       
                new_row = fci_stable_info
                writer.writerow(new_row)
                        
                del data, df_copied, fci_stable_info, new_row, output_fci_stable
                        
                gc.collect()
                        
                file2.flush()
                os.fdatasync(file2.fileno())
                        
                time_iter = time.time() - time_iter
              
                file1.write(f"DATA: {dataset_size}. DAG {j+1} of  {NUM_RANDOM_DAGS}. \n")
                file1.write(f"DATA: {dataset_size}. TimeIer: {time_iter :.3f}.AvgTime: {(time.time() - time0)/((j)+(i+1)) :.3f} seconds. \n")
                file1.flush()              # flush Python’s buffer
                os.fsync(file1.fileno())   # force OS to write to disk
                        
                       
                        
                    
                del  df
                gc.collect()
                
                        
                    
    
    file1.write(f"FINISHED. AvgTime: {(time.time() - time0)/(NUM_RANDOM_DAGS) :.3f} seconds. AvgTimeMarginal {time_marginal/NUM_RANDOM_DAGS:.3f} seconds. \n")
    file1.flush()              # flush Python’s buffer
    os.fsync(file1.fileno())   # force OS to write to disk
