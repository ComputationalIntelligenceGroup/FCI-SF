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

from causaldiscovery.algorithms.CSBS import csbs
from causaldiscovery.algorithms.PRCDSF import prcdsf
from causaldiscovery.algorithms.S_CDFSF import s_cdfsf
from causaldiscovery.algorithms.CSSU import cssu
from causaldiscovery.algorithms.LiNGAM_SF import lingam_sf
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


# Crear parser
parser = argparse.ArgumentParser(description="Ejecuta experimento con parámetros configurables.")

parser.add_argument("--numPVal", type=int, required=True, help="Number of pVal")
parser.add_argument("--numInstances", type=int, required=True, help="Data size")
parser.add_argument("--numVars", type=int, required=True, help="Number of variables")
parser.add_argument("--numDAGs", type=int, required=True, help="Number of random DAGs")
parser.add_argument("--numOrds", type=int, required=True, help="Number of random orders")


args = parser.parse_args()
# Asignar a variables
numPVal = args.numPVal
NUM_INSTANCES = args.numInstances
NUM_VARS = args.numVars
NUM_RANDOM_DAGS = args.numDAGs
NUM_ORDERS = args.numOrds

"""
numPVal = 0
NUM_INSTANCES = 375
NUM_VARS = 10
NUM_RANDOM_DAGS = 1
NUM_ORDERS = 1

"""

INITIAL_P = 0.1

NEIGHBORHOOD_SIZE = 2 
MAX_ITER = 1e3
CI_TEST = fisherz


NUM_DATASET_SIZES = 1



NUM_PERCENTAGE = 5
PERCENT_STEP = 0.20
percentList = [round((k+1)*PERCENT_STEP, 2) for k in range(NUM_PERCENTAGE)]

ALPHA = INITIAL_P/(2**numPVal)

time0 = time.time()


time_marginal = 0

time_iter = 0

# BASE RANDOM GENERATOR TO GUARANTEE REPRODUCIBILITY
base_rng = np.random.default_rng(123)


if NUM_VARS % NUM_PERCENTAGE != 0:
   raise AssertionError("NUM_VARS should be divisible by NUM_PERCENTAGE")

# Empty the file
file = open(f"../../../../logs/log_pVal{numPVal}_dataSize{NUM_INSTANCES}_nVars{NUM_VARS}_nbSize{NEIGHBORHOOD_SIZE}_nDAGs{NUM_RANDOM_DAGS}_nOrders{NUM_ORDERS}.txt", mode='w')
file.close()


with open(f"../../../../logs/log_pVal{numPVal}_dataSize{NUM_INSTANCES}_nVars{NUM_VARS}_nbSize{NEIGHBORHOOD_SIZE}_nDAGs{NUM_RANDOM_DAGS}_nOrders{NUM_ORDERS}.txt", "a", buffering=1) as file1:  # line-buffered in text mode

   
        
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
                
                digraph, df = make_bn_truth_and_sample(
                    base_rng = base_rng ,
                    num_vars = NUM_VARS,
                    expected_degree = NEIGHBORHOOD_SIZE,
                    n_samples = dataset_size
                )                
                            
                names = df.columns
             
                for i in range(0, NUM_ORDERS):
                    
                        time_iter = time.time()
        
                        random_permutation = np.random.permutation(NUM_VARS)
                        df_permuted = df.iloc[:, random_permutation]
                        
                        permuted_names = df_permuted.columns
                        data = df_permuted.to_numpy(copy=False)
                        df_permuted.rename(columns={permuted_names[i]: i for i in range(NUM_VARS)}, inplace=True)
                        
                        CI_test = myTest(df_permuted)
                        
                        output_csbs = (GeneralGraph([]), 0, 0, 0)
                        output_prcdsf = (GeneralGraph([]), 0, 0, 0)
                        output_scdfsf = (GeneralGraph([]), 0, 0, 0)
                        output_cssu = (GeneralGraph([]), 0, 0, 0)
                        output_lingam_sf = (GeneralGraph([]), 0, 0, 0)
                        output_fci_fs = (GeneralGraph([]), 0, 0, 0, [], {})
                        output_fci_stable = (GeneralGraph([]), 0, 0, 0, [], {})
                        
                        
                        csbs_info = []
                        prcdsf_info = []
                        scdfsf_info = []
                        cssu_info = []
                        lingam_sf_info = []
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
                            
                      
                            """
                            output_csbs = csbs_fs(data_marginal, independence_test_method=CI_test, alpha= ALPHA, initial_graph= output_csbs[0], new_node_names= new_names ,verbose = False, max_iter = MAX_ITER) 
                            output_prcdsf = prcdsf_fs(data_marginal, independence_test_method=CI_test, alpha= ALPHA,   initial_graph= output_prcdsf[0], new_node_names= new_names ,verbose = False, max_iter = MAX_ITER)
                            output_scdfsf = s_cdfsf_fs(data_marginal, independence_test_method=CI_test, alpha= ALPHA,   initial_graph= output_scdfsf[0], new_node_names= new_names ,verbose = False, max_iter = MAX_ITER)                    
                            output_cssu = cssu_fs(data_marginal, alpha= ALPHA,  initial_graph= output_cssu[0], new_node_names= new_names, verbose=False, max_iter= MAX_ITER)
                            output_fci_fs = fci_fs(data_marginal, independence_test_method=CI_test,  initial_sep_sets = output_fci_fs[5], alpha= ALPHA,  initial_graph= output_fci_fs[0] ,  new_node_names= new_names ,verbose = False)
                            output_fci_stable = fci_fs(data_marginal, independence_test_method=CI_test, initial_sep_sets = {}, alpha= ALPHA,  initial_graph = GeneralGraph([]), new_node_names = names_marginal, verbose = False)
                            """
                            
                         
                            # === Instrumentación alrededor de cada llamada ===
                            
                            
                            output_csbs = csbs(data_marginal, independence_test_method=CI_test, alpha= ALPHA, initial_graph= output_csbs[0], new_node_names= new_names ,verbose = False, max_iter = MAX_ITER) 
                            output_prcdsf = prcdsf(data_marginal, independence_test_method=CI_test, alpha= ALPHA,   initial_graph= output_prcdsf[0], new_node_names= new_names ,verbose = False, max_iter = MAX_ITER)
                            output_scdfsf = s_cdfsf(data_marginal, independence_test_method=CI_test, alpha= ALPHA,   initial_graph= output_scdfsf[0], new_node_names= new_names ,verbose = False, max_iter = MAX_ITER)                    
                            output_cssu = cssu(data_marginal, alpha= ALPHA,  initial_graph= output_cssu[0], new_node_names= new_names, verbose=False, max_iter= MAX_ITER)
                            output_lingam_sf = lingam_sf(data_marginal, independence_test_method=CI_test, alpha1= ALPHA, alpha2 = 0.001,  r = 0.7, initial_graph = output_lingam_sf[0], new_node_names= new_names,verbose = False, getMag = True  )
                            output_fci_fs = fci_sf(data_marginal, independence_test_method=CI_test,  initial_sep_sets = output_fci_fs[5], alpha= ALPHA,  initial_graph= output_fci_fs[0] ,  new_node_names= new_names ,verbose = False)
                            output_fci_stable = fci_sf(data_marginal, independence_test_method=CI_test, initial_sep_sets = {}, alpha= ALPHA,  initial_graph = GeneralGraph([]), new_node_names = names_marginal, verbose = False)
                         
                            
                            
                            file1.write(f"Percentage: {percentage}. ExecTimes:  csbs - {output_csbs[3]}. prcdsf - {output_prcdsf[3]}. scdfsf - {output_scdfsf[3]}. cssu - {output_cssu[3]}. LiNGAM-SF -  {output_lingam_sf[3]}. FCI-FS - {output_fci_fs[3]}. FCI - {output_fci_stable[3]}\n")
                            
                            #Metrics of the marginal models
                            csbs_info.extend(g_m.get_alg_marginal_info(ground_truth, output_csbs[0], output_csbs))
                            prcdsf_info.extend(g_m.get_alg_marginal_info(ground_truth, output_prcdsf[0], output_prcdsf))
                            scdfsf_info.extend(g_m.get_alg_marginal_info(ground_truth, output_scdfsf[0], output_scdfsf))
                            cssu_info.extend(g_m.get_alg_marginal_info(ground_truth, output_cssu[0], output_cssu))
                            lingam_sf_info.extend(g_m.get_alg_marginal_info(ground_truth, output_lingam_sf[0], output_lingam_sf))
                            fci_fs_info.extend(g_m.get_alg_marginal_info(ground_truth, output_fci_fs[0], output_fci_fs))
                            fci_stable_info.extend(g_m.get_alg_marginal_info(ground_truth, output_fci_stable[0], output_fci_stable))
                            
                          
                            
                        csbs_info.extend(g_m.get_self_comp_info(len(percentList), NUM_VARS, csbs_info))
                        prcdsf_info.extend(g_m.get_self_comp_info(len(percentList), NUM_VARS, prcdsf_info))
                        scdfsf_info.extend(g_m.get_self_comp_info(len(percentList), NUM_VARS, scdfsf_info))
                        cssu_info.extend(g_m.get_self_comp_info(len(percentList), NUM_VARS, cssu_info))
                        lingam_sf_info.extend(g_m.get_self_comp_info(len(percentList), NUM_VARS, lingam_sf_info))
                        fci_fs_info.extend(g_m.get_self_comp_info(len(percentList), NUM_VARS, fci_fs_info))
                        fci_stable_info.extend(g_m.get_self_comp_info(len(percentList), NUM_VARS, fci_stable_info))
                        
                       
                        new_row = csbs_info + prcdsf_info + scdfsf_info + lingam_sf_info + cssu_info + fci_fs_info + fci_stable_info
                        writer.writerow(new_row)
                        
                        del data, df_permuted, csbs_info, prcdsf_info, scdfsf_info, cssu_info, lingam_sf_info, fci_fs_info, fci_stable_info, new_row, output_csbs, output_prcdsf, output_scdfsf, output_cssu, output_lingam_sf, output_fci_fs, output_fci_stable
                        
                        gc.collect()
                        
                        file2.flush()
                        os.fdatasync(file2.fileno())
                        
                        time_iter = time.time() - time_iter
              
                        file1.write(f"DATA: {dataset_size}. DAG {j+1} of  {NUM_RANDOM_DAGS}. ORDER {i+1} of {NUM_ORDERS}. \n")
                        file1.write(f"DATA: {dataset_size}. TimeIer: {time_iter :.3f}.AvgTime: {(time.time() - time0)/((j*NUM_ORDERS)+(i+1)) :.3f} seconds. AvgTimeMarginal {time_marginal/((j*NUM_ORDERS)+(i+1)):.3f} seconds. \n")
                        file1.flush()              # flush Python’s buffer
                        os.fsync(file1.fileno())   # force OS to write to disk
                        
                       
                        
                    
                del  df
                gc.collect()
                
                        
                    
    
    file1.write(f"FINISHED. AvgTime: {(time.time() - time0)/(NUM_RANDOM_DAGS*NUM_ORDERS) :.3f} seconds. AvgTimeMarginal {time_marginal/NUM_RANDOM_DAGS:.3f} seconds. \n")
    file1.flush()              # flush Python’s buffer
    os.fsync(file1.fileno())   # force OS to write to disk
