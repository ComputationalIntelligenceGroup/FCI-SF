# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 19:56:41 2025

@author: chdem
"""

from pgmpy.estimators.CITests import *
import pandas as pd
import numpy as np
from causallearn.utils.cit import CIT_Base

class myTest:
    
    
    
    def __init__(self, data, **kwargs):
        self.data = data
    
    def __call__(self, X, Y, condition_set=None):
        if condition_set is not None:
            condition_set  = list(condition_set)
        else:
            condition_set = []
            
        return g_sq(X=X, Y=Y, Z=condition_set, data=self.data, boolean=False)[1]