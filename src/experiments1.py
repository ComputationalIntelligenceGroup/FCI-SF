#!/usr/bin/env python3
"""
experiments1
"""



# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 19:26:47 2025

@author: chdem
"""

from pgmpy.models import BayesianNetwork


import csv
import numpy as np
import time

from datetime import datetime

    
    

    


numPVal =  1

filePath = f"../log{numPVal}.txt"

fdLog = open(filePath, "w")



now = datetime.now()
print(f"\nNew execution. P-value: {numPVal}. Date: {now}.\n", file=fdLog)

ALGORITHMS = [ "CSBS", "PRCDSF", "SCDFSF", "CSSU", "FCIFS"]

GRAPHICAL_SCORE_TYPES = ["e", "a", "c", "t"]

INITIAL_P = 0.1

NUM_VARS = 100

NEIGHBORHOOD_SIZE = 3

INITIAL_COLUMNS = ["random_seed"]

METRICS_NT = ["numCI", "execTime", "avgSepSize", "HD", "SHDe", "SHD"]

METRICS_T = ["TP", "FP", "TN", "FN", "PREC", "RECALL", "F1"]

METRICS_SC = ["SCHD", "SCHDn", "SCSHDe", "SCSHDen", "SCSHD", "SCSHDn"]


NUM_INSTANCES = 500


NUM_RANDOM_DAGS = 1

NUM_ORDERS = 1

NUM_PERCENTAGE = 5

PERCENT_STEP = 0.20

percentList = [round((k+1)*PERCENT_STEP, 2) for k in range(NUM_PERCENTAGE) ]


p = INITIAL_P/(2**numPVal)

with open(f"output{numPVal}.csv", mode='w', newline='') as file:
    writer = csv.writer(file)
    #Write a header row
    
    column_names = []
    
    for initial_column in INITIAL_COLUMNS:
        column_names.append(initial_column)
    
    for alg_name in ALGORITHMS:
        for percentage in percentList:
            suffix = "_" + alg_name + "_" + str(int(percentage*100))
            
            for metric_nt in METRICS_NT:
                column_name = metric_nt + suffix
                column_names.append(column_name)
                
            for metric_type in GRAPHICAL_SCORE_TYPES:
                suffix2 = suffix + "_" + metric_type
                
                for metric_t in METRICS_T:
                    column_name = metric_t + suffix2
                    column_names.append(column_name)
                    
        suffix = "_" + alg_name 
        for metric_sc in METRICS_SC:
            column_name = metric_sc + suffix
            column_names.append(column_name)
            
            
                    
        
                    
                    
                    
                    
                
                
            
    
    
    writer.writerow(column_names)
    for j in range(0, NUM_RANDOM_DAGS):
        
        row = []
            
        n_states = np.random.randint(low=3, high=6, size=NUM_VARS)
        #Generate random DAG
        model = BayesianNetwork.get_random(n_nodes=NUM_VARS, edge_prob=NEIGHBORHOOD_SIZE/(NUM_VARS-1), n_states = n_states)
            
            
         # Simulate data

        data = model.simulate(n_samples=NUM_INSTANCES)
        variables = list(data.columns)
            
        

      
            
        PDAG_stable, nCITestStable =  estFS.estimate(variant = "MPC-stable", ci_test='chi_square', max_cond_vars=3,  significance_level= p, new_vars = variables)
            
        
            
        totalTimeStable = timeStable1 - timeStable0
            
        stable_metrics = PDAG_stable.getGraphicalMetrics(PDAG_base)
            
        stable_metrics =  stable_metrics + [nCITestStable, totalTimeStable ]
            
            
        
        for i in range(0, NUM_ORDERS):

                data = random_var_order(data) 
                
                estFS = PCFS(data) #PC-FS
                variables = list(data.columns)
                
                timeOrig0 = time.time()
                
                PDAG_orig, numCIOrig =  estFS.estimate(variant = "orig", ci_test='chi_square', max_cond_vars=3,  significance_level= p, new_vars = variables)
                
                timeOrig1 = time.time()
                
                totalTimeOrig = timeOrig1 - timeOrig0
                
                
                orig_metrics = PDAG_orig.getGraphicalMetrics(PDAG_base)
                
                orig_metrics = orig_metrics + [numCIOrig, totalTimeOrig]
                
                fs_metrics = []
                
                for percentage in percentList :
                
                    newVars, initialVars = extract_random(data, percentage = percentage)
                    
                    timeMarginal0 = time.time()
                    
                    PDAG_marginal, numCIMarginal = estFS.estimate(variant = "MPC-stable", ci_test='chi_square', max_cond_vars=3,  significance_level=p, new_vars = initialVars)
                   
                    timeMarginal1 = time.time()
                   
                    totalTimeMarginal = timeMarginal1 - timeMarginal0
                    
                    timeFS0 = time.time()
                    
                    PDAG_FS, numCIFS = estFS.estimate(variant = "MPC-stable", ci_test='chi_square', max_cond_vars=3, pdag = PDAG_marginal,   significance_level=p, new_vars = newVars)
                    
                    timeFS1 = time.time()
                    
                    totalTimeFS = timeFS1 - timeFS0
                    
                    
                    
                    fs_metrics = fs_metrics + PDAG_FS.getGraphicalMetrics(PDAG_base) + [numCIMarginal, totalTimeMarginal, numCIFS, totalTimeFS]
                
                
                newRow = orig_metrics + stable_metrics + fs_metrics
                print(f"\nSIMULATE DATA: {numPVal} - {j+1}:{NUM_RANDOM_DAGS} - {i+1}:{NUM_ORDERS}.\n", file=fdLog)
                
                writer.writerow(newRow)

print(f"Finished. P-value. {numPVal}", file=fdLog)