# -*- coding: utf-8 -*-
"""
realworld experiments
"""
import csv
import numpy as np
import time

import sys, pathlib
repo_root = pathlib.Path(__file__).resolve().parents[3]

print(repo_root)  
sys.path.append(str(repo_root / "src" / "external" / "causal-self-compatibility/src" ))

sys.path.append(str(repo_root / "src"))

from self_compatibility import SelfCompatibilityScorer
from causal_graphs.pag import  PAG

import warnings
warnings.filterwarnings("error", message="The truth value of an empty array is ambiguous*")

from causallearn.graph.GeneralGraph import GeneralGraph 

from causaldiscovery.algorithms.CSBS import csbs
from causaldiscovery.algorithms.PRCDSF import prcdsf
from causaldiscovery.algorithms.S_CDFSF import s_cdfsf
from causaldiscovery.algorithms.CSSU import cssu
from causaldiscovery.algorithms.FCI_SF import fci_sf
from causaldiscovery.CItest.noCache_CI_Test import myTest

import os


# Load the data

import pandas as pd
from glob import glob

NUM_ORDERS = 1
ALPHA = 0.05
MAX_ITER = 1e3
NUM_ALGO = 5

NUM_PERCENTAGE = 5
PERCENT_STEP = 0.20
percentList = [round((k+1)*PERCENT_STEP, 2) for k in range(NUM_PERCENTAGE)]


# Ruta a la carpeta con los CSV
ruta = r"../clean_datasets/real-world_datasets/*.csv"

# DataFrames list
dfs = [pd.read_csv(fichero) for fichero in glob(ruta)]

# Prepare algorithms

output_csbs = (GeneralGraph([]), 0, 0, 0)
output_prcdsf = (GeneralGraph([]), 0, 0, 0)
output_scdfsf = (GeneralGraph([]), 0, 0, 0)
output_cssu = (GeneralGraph([]), 0, 0, 0)
output_fci_fs = (GeneralGraph([]), 0, 0, 0, [], {})
output_fci_stable = (GeneralGraph([]), 0, 0, 0, [], {})

# Use the data to learn the graphs

# Empty the file
file = open("../../logs/real-world-datasets.txt", mode='w')
file.close()


results_matrix = np.zeros((len(dfs), NUM_ALGO))

time_initial = time.time()

pos_elim = [2,5,7,9,11, 13]

dfs = [item for pos, item in enumerate(dfs) if pos not in pos_elim]

with open("../../logs/real-world-datasets.txt", "a", buffering=1) as file1:  # line-buffered in text mode

    for j in range(len(dfs)):
        df = dfs[6]
        names = df.columns
        num_vars = len(names)
        kgs = results_matrix[j]
        time_order = time.time()
        
        for i in range(0, NUM_ORDERS):
            
            random_permutation = np.random.permutation(num_vars)
            df_permuted = df.iloc[:, random_permutation].copy()
            
            permuted_names = df_permuted.columns
            data = df_permuted.to_numpy(copy=False)
            df_permuted.rename(columns={permuted_names[i]: i for i in range(num_vars)}, inplace=True)
            
            CI_test = myTest(df_permuted)
            
            # First, learn the full graph
            
            print("Starting full")
         
            csbs_full = csbs(data, independence_test_method=CI_test, alpha= ALPHA, initial_graph= GeneralGraph([]), new_node_names= permuted_names ,verbose = False, max_iter = MAX_ITER) 
            
            print("csbs full done")
            
            prcdsf_full = prcdsf(data, independence_test_method=CI_test, alpha= ALPHA,   initial_graph= GeneralGraph([]), new_node_names= permuted_names ,verbose = False, max_iter = MAX_ITER)
            
            print("prcdsf_full full done")
            
            
            scdfsf_full = s_cdfsf(data, independence_test_method=CI_test, alpha= ALPHA,   initial_graph= GeneralGraph([]), new_node_names= permuted_names ,verbose = False, max_iter = MAX_ITER)                    
        
            print("scdfsf_full full done")
            
            cssu_full = cssu(data, alpha= ALPHA,  initial_graph= GeneralGraph([]), new_node_names= permuted_names, verbose=False, max_iter= MAX_ITER)
         
            print("cssu_full full done")
            
            fci_stable_full = fci_sf(data, independence_test_method=CI_test, initial_sep_sets = {}, alpha= ALPHA,  initial_graph = GeneralGraph([]), new_node_names = permuted_names, verbose = False)
            
          
            
            full_graphs = [  PAG(csbs_full[0]), PAG(prcdsf_full[0]), PAG(scdfsf_full[0]), PAG(cssu_full[0]), PAG(fci_stable_full[0])]
            
            csbs_graphs = []
            prcdsf_graphs = []
            scdfsf_graphs = []
            cssu_graphs = []
            fci_fs_graphs = []
            
            output_csbs = (GeneralGraph([]), 0, 0, 0)
            output_prcdsf = (GeneralGraph([]), 0, 0, 0)
            output_scdfsf = (GeneralGraph([]), 0, 0, 0)
            output_cssu = (GeneralGraph([]), 0, 0, 0)
            output_fci_fs = (GeneralGraph([]), 0, 0, 0, [], {})

            
            print("Full graphs ok")
            
            for percentage in percentList :
                
                
                start_pos =int(round((percentage - PERCENT_STEP)*num_vars))
                end_pos = int(round(percentage*num_vars))
                
                data_marginal = data[:, 0:end_pos]
                names_marginal = permuted_names[0: end_pos]
                names_latent = permuted_names[end_pos : ]
                new_names = permuted_names[ start_pos:end_pos]
               
                output_csbs = csbs(data_marginal, independence_test_method=CI_test, alpha= ALPHA, initial_graph= output_csbs[0], new_node_names= new_names ,verbose = False, max_iter = MAX_ITER) 
                output_prcdsf = prcdsf(data_marginal, independence_test_method=CI_test, alpha= ALPHA,   initial_graph= output_prcdsf[0], new_node_names= new_names ,verbose = False, max_iter = MAX_ITER)
                output_scdfsf = s_cdfsf(data_marginal, independence_test_method=CI_test, alpha= ALPHA,   initial_graph= output_scdfsf[0], new_node_names= new_names ,verbose = False, max_iter = MAX_ITER)                    
                output_cssu = cssu(data_marginal, alpha= ALPHA,  initial_graph= output_cssu[0], new_node_names= new_names, verbose=False, max_iter= MAX_ITER)
               
                
                output_fci_fs = fci_sf(data_marginal, independence_test_method=CI_test,  initial_sep_sets = output_fci_fs[5], alpha= ALPHA,  initial_graph= output_fci_fs[0] ,  new_node_names= new_names ,verbose = False)
                
                file1.write(f"Percentage: {percentage}. ExecTimes: scdfsf - {output_scdfsf[3]}. cssu - {output_cssu[3]}. FCI-FS - {output_fci_fs[3]}. \n")
                file1.flush()              # flush Python’s buffer
                os.fsync(file1.fileno())   # force OS to write to disk
                
                csbs_graphs.append(PAG(output_csbs[0])) 
                prcdsf_graphs.append(PAG(output_prcdsf[0]))
                scdfsf_graphs.append(PAG(output_scdfsf[0]))
                cssu_graphs.append(PAG(output_cssu[0]))
               
                fci_fs_graphs.append(PAG(output_fci_fs[0]))
                
            
            subset_graphs = [csbs_graphs, prcdsf_graphs, scdfsf_graphs, cssu_graphs, fci_fs_graphs]
                
            
            scorer = SelfCompatibilityScorer(None, len(percentList))
            
            to_eval = list(zip(full_graphs, subset_graphs))
         
            
           
            
            for k in range(NUM_ALGO):
                
                eval_graphs = to_eval[k]
                
                time_g0 = time.time()
                # Graphical metric
                kgs[k] += scorer._graphical_compatibility(eval_graphs[0], eval_graphs[1])
                
                file1.write(f"Time for graphical self-compatibility {k} data {j}:  {time.time() - time_g0}\n")
                file1.flush()              # flush Python’s buffer
                os.fsync(file1.fileno())   # force OS to write to disk
                
        kgs[:] = [ (1/NUM_ORDERS) * x for x in kgs ]
        
        file1.write(f"Time dataset {j}: {time.time() - time_order}\n")
        file1.flush()              # flush Python’s buffer
        os.fsync(file1.fileno())   # force OS to write to disk
        
    file1.write(f"Time all datasets: {time.time() - time_initial}\n")
    file1.flush()              # flush Python’s buffer
    os.fsync(file1.fileno())   # force OS to write to disk


with open("../../logs/results_matrix.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(results_matrix)

               
           






# Make critical inference diagram

