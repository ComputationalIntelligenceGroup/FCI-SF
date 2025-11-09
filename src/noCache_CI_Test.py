# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 19:56:41 2025

@author: chdem
"""

from pgmpy.estimators.CITests import *
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

from numba import njit


class myTest:
    
    def __init__(self, data, **kwargs):
        self.data = data
        
        

   
    
    def __call__(self, X, Y, condition_set=None):
        if condition_set is not None:
            condition_set  = list(condition_set)
        else:
            condition_set = []

        r, p = fast_partial_corr_jit_df(self.data, X, Y, condition_set)
          
        return p
    


def fast_partial_corr_jit_df(df, x_col, y_col, z_cols):
    """
    Compute partial correlation r and p-value between x_col and y_col
    given conditioning columns z_cols, using JIT-accelerated code.
    """
    X = df[x_col].values
    Y = df[y_col].values
    Z = df[z_cols].values if z_cols else np.empty((len(X), 0))
    return _fast_partial_corr_jit(X, Y, Z)

@njit
def _fast_partial_corr_jit(X, Y, Z):
    n = X.shape[0]
    k = Z.shape[1]

    if k == 0:
        # Just Pearson correlation
        r = np.dot(X, Y) / (np.linalg.norm(X) * np.linalg.norm(Y))
    else:
        ZTZ_inv = np.linalg.inv(Z.T @ Z)
        Z_pinv = ZTZ_inv @ Z.T
        X_res = X - Z @ (Z_pinv @ X)
        Y_res = Y - Z @ (Z_pinv @ Y)
        r = np.dot(X_res, Y_res) / (np.linalg.norm(X_res) * np.linalg.norm(Y_res))

    # Fisher-style t-statistic and normal approximation
    df = n - k - 2
    t = r * np.sqrt(df / (1 - r**2))
    z = np.abs(t)
    p = 2 * (1 - math_erf(z / np.sqrt(2)))
    return r, p

@njit
def math_erf(x):
    # Abramowitz & Stegun rational approximation of erf(x)
    sign = 1 if x >= 0 else -1
    x = abs(x)
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5*t + a4)*t + a3)*t + a2)*t + a1)*t) * np.exp(-x*x)
    return sign * y
