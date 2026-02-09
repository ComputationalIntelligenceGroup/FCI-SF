# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 16:54:29 2026

@author: chdem
"""
import numpy as np

def _rbf_kernel(v: np.ndarray, sigma: float | None = None, eps: float = 1e-12) -> np.ndarray:
    """
    RBF kernel matrix for 1D vector v (n,)
    Uses median heuristic if sigma is None.
    """
    v = v.reshape(-1, 1)
    d2 = (v - v.T) ** 2  # squared distances

    if sigma is None:
        # median heuristic on non-zero distances
        tri = d2[np.triu_indices_from(d2, k=1)]
        med = np.median(tri[tri > 0]) if np.any(tri > 0) else 1.0
        sigma = np.sqrt(med + eps)

    gamma = 1.0 / (2.0 * (sigma**2) + eps)
    return np.exp(-gamma * d2)


def hsic_statistic(x: np.ndarray, y: np.ndarray) -> float:
    """
    Biased HSIC estimator with RBF kernels.
    Returns scalar HSIC value.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n = x.shape[0]
    if n < 4:
        return 0.0

    K = _rbf_kernel(x)
    L = _rbf_kernel(y)

    H = np.eye(n) - np.ones((n, n)) / n
    Kc = H @ K @ H
    Lc = H @ L @ H
    # biased HSIC:
    return float(np.trace(Kc @ Lc) / (n * n))