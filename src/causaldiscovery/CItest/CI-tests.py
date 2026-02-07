import os, json, codecs, time, hashlib
import numpy as np
from math import log, sqrt
from collections.abc import Iterable
from scipy.stats import chi2, norm

from causallearn.utils.KCI.KCI import KCI_CInd, KCI_UInd
from causallearn.utils.FastKCI.FastKCI import FastKCI_CInd, FastKCI_UInd
from causallearn.utils.RCIT.RCIT import RCIT as RCIT_CInd
from causallearn.utils.RCIT.RCIT import RIT as RCIT_UInd
from causallearn.utils.PCUtils import Helper
from causallearn.utils.cit import CIT_Base, NO_SPECIFIED_PARAMETERS_MSG

class FisherZ(CIT_Base):
    def __init__(self, data, correlation_matrix,  **kwargs):
        super().__init__(data, **kwargs)
        self.assert_input_data_is_valid()
        self.correlation_matrix = correlation_matrix

    def __call__(self, X, Y, condition_set=None):
        '''
        Perform an independence test using Fisher-Z's test.

        Parameters
        ----------
        X, Y and condition_set : column indices of data

        Returns
        -------
        p : the p-value of the test
        '''
        
        if condition_set is None: condition_set_aux = []
        # 'int' to convert np.*int* to built-in int; 'set' to remove duplicates; sorted for hashing
        condition_set_aux = sorted(set(map(int, condition_set)))
        
        X_aux, Y_aux = (int(X), int(Y)) if (X < Y) else (int(Y), int(X))
        assert X_aux not in condition_set and Y_aux not in condition_set, "X, Y cannot be in condition_set."
        Xs, Ys = [X], [Y]
        
        var = Xs + Ys + condition_set_aux
        sub_corr_matrix = self.correlation_matrix[np.ix_(var, var)]
        try:
            inv = np.linalg.inv(sub_corr_matrix)
        except np.linalg.LinAlgError:
            raise ValueError('Data correlation matrix is singular. Cannot run fisherz test. Please check your data.')
        r = -inv[0, 1] / sqrt(abs(inv[0, 0] * inv[1, 1]))
        if abs(r) >= 1: r = (1. - np.finfo(float).eps) * np.sign(r) # may happen when samplesize is very small or relation is deterministic
        Z = 0.5 * log((1 + r) / (1 - r))
        X = sqrt(self.sample_size - len(condition_set_aux) - 3) * abs(Z)
        p = 2 * (1 - norm.cdf(abs(X)))
  
        return p