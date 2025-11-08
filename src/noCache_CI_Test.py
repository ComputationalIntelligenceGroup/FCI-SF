# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 19:56:41 2025

@author: chdem
"""

from pgmpy.estimators.CITests import *
import pandas as pd
import numpy as np
from scipy.stats import pearsonr


class myTest:
    
    
    
    
    
    def __init__(self, data, **kwargs):
        self.data = data
        
        

   
    
    def __call__(self, X, Y, condition_set=None):
        if condition_set is not None:
            condition_set  = list(condition_set)
        else:
            condition_set = []

        r, p = fisher_z_from_df(self.data, X, Y, condition_set)
          
        return p
    
def fisher_z_from_df(df, x, y, cond):
       X = df[x].values
       Y = df[y].values
       Z = df[cond].values if len(cond) > 0 else None
   
       if Z is None or Z.shape[1] == 0:
           r, p = pearsonr(X, Y)
       else:
           # Regress X ~ Z and Y ~ Z
           beta_X, *_ = np.linalg.lstsq(Z, X, rcond=None)
           beta_Y, *_ = np.linalg.lstsq(Z, Y, rcond=None)
           res_X = X - Z @ beta_X
           res_Y = Y - Z @ beta_Y
           r, p = pearsonr(res_X, res_Y)
   
       return r, p
