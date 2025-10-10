#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np



def gaussian_entropy(cov, base: float =None):
    """cov: (d,d) covariance matrix."""
        
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        raise ValueError("Covariance must be PD.")
    d = cov.shape[0]
    h = 0.5 * (d * np.log(2*np.pi*np.e) + logdet)
    return h if base is None else h * 1/(np.log(base))

def gaussian_mi_from_cov_blocks(Sxx, Syy, Sxy, base: float =None):
    """Mutual information for jointly Gaussian X,Y given block covariances."""
    # Use log-det ratio form for stability
    sign_x, logdet_x = np.linalg.slogdet(Sxx)
    sign_y, logdet_y = np.linalg.slogdet(Syy)
    # joint det via Schur complement: det(S) = det(Sxx) * det(Syy - S_yx Sxx^{-1} S_xy)
    Sxx_inv = np.linalg.inv(Sxx)
    Schur = Syy - Sxy.T @ Sxx_inv @ Sxy
    sign_s, logdet_s = np.linalg.slogdet(Schur)
    if sign_x <= 0 or sign_y <= 0 or sign_s <= 0:
        raise ValueError("Covariance not positive definite.")
    logdet_joint = logdet_x + logdet_s
    I = 0.5 * (logdet_x + logdet_y - logdet_joint)
    return I if base is None  else I * 1/(np.log(base))


def gaussian_su_from_samples(X, Y, base: float=None):
    """
    X: (N, d_x), Y: (N, d_y)
    Computes SU assuming joint Gaussianity.
    """
    
    X = np.asarray(X)
    Y = np.asarray(Y)
    # ensure 2D column shape
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    Z = np.hstack([X, Y])
    cov = np.cov(Z, rowvar=False, bias=False)
    d_x = X.shape[1] if X.ndim > 1 else 1
    Sxx = cov[:d_x, :d_x]
    Syy = cov[d_x:, d_x:]
    Sxy = cov[:d_x, d_x:]
    I = gaussian_mi_from_cov_blocks(Sxx, Syy, Sxy, base=base)
    hX = gaussian_entropy(Sxx, base=base)
    hY = gaussian_entropy(Syy, base=base)
    return 2 * I / (hX + hY)
