#!/usr/bin/env python3
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

import time 

numPVal =  1
filePath = f"../log{numPVal}.txt"
fdLog = open(filePath, "w")

now = datetime.now()
print(f"\nNew execution. P-value: {numPVal}. Date: {now}.\n", file=fdLog)

INITIAL_P = 0.1
NUM_VARS = 50
NEIGHBORHOOD_SIZE = 1.5
MAX_ITER = 1e3
CI_TEST = fisherz


NUM_DATASET_SIZES = 1
NUM_INSTANCES = 10000
NUM_RANDOM_DAGS = 1
NUM_ORDERS = 1
NUM_PERCENTAGE = 5
PERCENT_STEP = 0.20
percentList = [round((k+1)*PERCENT_STEP, 2) for k in range(NUM_PERCENTAGE)]

ALPHA = INITIAL_P/(2**numPVal)

time0 = time.time()

with open(f"output{numPVal}.csv", mode='w', newline='') as file:
    writer = csv.writer(file)
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
        
        for tam_mult in range(0, NUM_DATASET_SIZES):
            dataset_size = NUM_INSTANCES*(2**tam_mult)
        
            gen = AcyclicGraphGenerator('linear', npoints=NUM_INSTANCES, nodes=NUM_VARS, dag_type='erdos', expected_degree= NEIGHBORHOOD_SIZE)
            df, digraph = gen.generate()   # X: DataFrame, G: networkx.DiGraph
                        
            names = df.columns
            
            full_ground_truth = to_PAG(digraph, names)
            
          
            
            
        
            
            for i in range(0, NUM_ORDERS):
    
                    random_permutation = np.random.permutation(NUM_VARS)
                    df_permuted = df.iloc[:, random_permutation]
                    df_permuted.to_csv(f"../experiments_data/Data-DAG{j}-Size{dataset_size}-Order{i}.csv", index=False, float_format="%.15g")
                    
                    permuted_names = df_permuted.columns
                    data = df_permuted.values
                    
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
                        
                        print(f"Percentage: {percentage}")
                        
                        start_pos =int((percentage - PERCENT_STEP)*NUM_VARS)
                        
                        
                        
                        end_pos = int(percentage*NUM_VARS)
                        data_marginal = data[:, 0:end_pos]
                        names_marginal = permuted_names[0: end_pos]
                        names_latent = permuted_names[end_pos : ]
                        new_names = permuted_names[ start_pos:end_pos]
                        
                        print("Everything ok")
                        
                        ground_truth = dag2pag(digraph, names_marginal)
                        
                        print("Going for csbs")
                        
                        output_csbs = csbs_fs(data_marginal, independence_test_method=CI_TEST, alpha= ALPHA, initial_graph= output_csbs[0], new_node_names= new_names ,verbose = False, max_iter = MAX_ITER) 
                        
                        print(f"Percentage: {percentage} csbs done")
                        
                        output_prcdsf = prcdsf_fs(data_marginal, independence_test_method=CI_TEST, alpha= ALPHA,   initial_graph= output_prcdsf[0], new_node_names= new_names ,verbose = False, max_iter = MAX_ITER)
                        
                        print(f"Percentage: {percentage} prcdsf done")
                        
                        output_scdfsf = s_cdfsf_fs(data_marginal, independence_test_method=CI_TEST, alpha= ALPHA,   initial_graph= output_scdfsf[0], new_node_names= new_names ,verbose = False, max_iter = MAX_ITER)                    
                        
                        print(f"Percentage: {percentage} scdfsf done")
                        
                        output_cssu = cssu_fs(data_marginal, alpha= ALPHA,  initial_graph= output_cssu[0], new_node_names= new_names, verbose=False, max_iter= MAX_ITER)
                        
                        print(f"Percentage: {percentage} cssu done")
                        
                        output_fci_fs = fci_fs(data_marginal, independence_test_method=CI_TEST,  initial_sep_sets = output_fci_fs[5], alpha= ALPHA,  initial_graph= output_fci_fs[0] ,  new_node_names= new_names ,verbose = False)
                        output_fci_stable = fci_fs(data_marginal, independence_test_method=CI_TEST, initial_sep_sets = {}, alpha= ALPHA,  initial_graph = GeneralGraph([]), new_node_names = names_marginal, verbose = False)
                        
                        print(f"Percentage: {percentage} train done")
                        
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
                    
            

            print(f"DAG {j} of  {NUM_RANDOM_DAGS}. ORDER {i} of {NUM_ORDERS}.")
                
print(f"Time: {time.time() - time0:.3f} seconds")
print(f"Finished. P-value: {numPVal}.", file=fdLog)