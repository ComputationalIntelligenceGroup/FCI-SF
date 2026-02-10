# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 16:46:10 2026

@author: chdem
"""

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy.stats import ttest_1samp


# -------------------------
# Whitening (like MATLAB SVD)
# -------------------------
def pca_whiten_full(X, eps=0.1):
    """
    X: (n_features, n_samples)  (MATLAB-style: rows=variables)
    Returns:
      PCAWhite: (n_features, n_samples)
      U: (n_features, n_features)
      S: (n_features,) eigenvalues
    """
    n, m = X.shape
    Xc = X - X.mean(axis=1, keepdims=True)

    sigma = (1.0 / m) * (Xc @ Xc.T)  # covariance
    # symmetric eigendecomp (equiv to svd on covariance)
    S, U = np.linalg.eigh(sigma)
    idx = np.argsort(S)[::-1]
    S = S[idx]
    U = U[:, idx]
    
    # Numerical safety: clip negative eigenvalues to 0
    S = np.clip(S, 0.0, None)

    PCAWhite = (np.diag(1.0 / np.sqrt(S + eps)) @ (U.T @ Xc))
    PCAWhite = np.nan_to_num(PCAWhite, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return PCAWhite, U.astype(np.float32), S.astype(np.float32)


# -------------------------
# RICA model
# -------------------------
class RICA(nn.Module):
    """
    Learn Z (n_features x n_components).
    For whitened sample row w (1 x n_features):
      h = w @ Z
      w_hat = h @ Z^T = w @ Z @ Z^T
    """
    def __init__(self, n_features, n_components, seed=0):
        super().__init__()
        g = torch.Generator()
        g.manual_seed(seed)
        Z = torch.randn(n_features, n_components, generator=g) * 0.01
        self.Z = nn.Parameter(Z)

    def forward(self, W):
        H = W @ self.Z
        W_hat = H @ self.Z.T
        return H, W_hat


def rica_fit_transformweights(
    X_white_samples_by_features,
    n_components,
    batch_size,
    lam=0.5,
    sparsity="logcosh",
    lr=1e-2,
    epochs=1500,
    seed=0,
    device=None,
    verbose=False
):
    """
    Memory-safe RICA training using full-batch Adam (LBFGS removed).
    This is the most practical substitute for MATLAB rica when called many times.

    X_white_samples_by_features: (n_samples, n_features)
    Returns:
      Z_star: (n_features, n_components)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(seed)
    W = torch.tensor(X_white_samples_by_features, dtype=torch.float32, device=device)

    n_samples, n_features = W.shape
    model = RICA(n_features, n_components, seed=seed).to(device)

    # Adam is stable + light on memory; full-batch is MATLAB-like.
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def sparsity_penalty(H):
        if sparsity == "l1":
            return H.abs().mean()
        if sparsity == "logcosh":
            # stable logcosh: |x| + log(1 + exp(-2|x|)) - log(2)
            absH = torch.abs(H)
            return (absH + torch.log1p(torch.exp(-2.0 * absH)) - np.log(2.0)).mean()
        raise ValueError("sparsity must be 'logcosh' or 'l1'")

    # If batch_size is None/0/too big, do full batch (recommended here)
    if not batch_size or batch_size >= n_samples:
        batch_size = n_samples

    rng = np.random.default_rng(seed)

    for ep in range(1, epochs + 1):
        idx = rng.permutation(n_samples)

        total_loss = 0.0
        for start in range(0, n_samples, batch_size):
            bidx = idx[start:start + batch_size]
            Wb = W[bidx]

            H, W_hat = model(Wb)
            sp = sparsity_penalty(H)
            rec = ((W_hat - Wb) ** 2).mean()
            loss = sp + lam * rec

            opt.zero_grad()
            loss.backward()

            # Gradient clipping helps avoid NaNs/infs and keeps training stable
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            opt.step()

            total_loss += loss.item() * len(bidx)

        if verbose and (ep % 50 == 0 or ep == 1 or ep == epochs):
            print(f"epoch {ep:4d} | loss {total_loss / n_samples:.6f}")

        # Optional early stop if it stabilizes (saves lots of time)
        # You can remove this if you want exact fixed epochs.
        if ep > 50 and (total_loss / n_samples) < 1e-6:
            break

    Z_star = model.Z.detach().cpu().numpy().astype(np.float32)
    return Z_star




# -------------------------
# Identify_Zero (MATLAB -> Python)
# -------------------------
def identify_zero(samples_rs, l_num, thresh=0.5, alpha=0.05):
    """
    samples_rs: (n_features, n_runs*l_num) concatenated weights, each run is a block of l_num columns
    l_num: number of components per run
    Returns:
      support: (n_features, l_num) binary mask
    """
    w, l = samples_rs.shape
    assert l % l_num == 0, "samples_rs columns must be multiple of l_num"
    n_runs = l // l_num

    first = samples_rs[:, 0:l_num]  # reference run
    statisc_rows = [first.reshape(1, w * l_num)]

    # Align each subsequent run to the first
    for run in range(1, n_runs):
        second = samples_rs[:, run * l_num:(run + 1) * l_num]

        # distance between components: compare columns (components)
        # MATLAB: pdist2(first', second') -> shapes (l_num, l_num)
        dis = cdist(first.T, second.T, metric="euclidean")

        # Hungarian assignment: minimize total distance
        row_ind, col_ind = linear_sum_assignment(dis)
        # row_ind should be [0..l_num-1], col_ind gives best matching columns in 'second'
        second_new = second[:, col_ind]

        statisc_rows.append(second_new.reshape(1, w * l_num))

    statisc_value = np.vstack(statisc_rows)  # (n_runs, w*l_num)

    # binarize abs value at threshold
    statisc_value = np.abs(statisc_value)
    statisc_value = (statisc_value >= thresh).astype(np.float32)

    # t-test each entry vs 0
    # MATLAB ttest(x,0)==1 means reject mean==0 at default alpha (0.05)
    t_stat, p_vals = ttest_1samp(statisc_value, popmean=0.0, axis=0, alternative="greater")
    weight_flat = (p_vals < alpha).astype(np.int32)

    support = weight_flat.reshape(w, l_num)
    return support


# -------------------------
# orientation (MATLAB -> Python)
# -------------------------
def orientation(
    data,
    r,
    runs=5,
    eps=0.1,
    lam=0.5,
    sparsity="logcosh",
    lr=1e-1,
    batch_size=None,
    epochs=1000           ,
    seed=0,
    device=None,
    verbose = False,
    thresh=0.5,
    alpha=0.05
):
    """
    data: ndarray, assumed shape (n_samples, n_features) OR (n_features, n_samples).
          MATLAB code does X = data' then treats rows=features.
    r: sampling ratio (0 < r <= 1), sample_num = round(m*r)

    Returns:
      support: (n_features, n_components) binary mask (like Identify_Zero output)
      weights_concat: (n_features, runs*n_components) numeric weights from each run (for debugging)
    """
    X = np.asarray(data, dtype=np.float32)
    
    # Match MATLAB convention: X = data' so X is (n_features, n_samples)
    if X.shape[0] > X.shape[1]:
        # ambiguous; we choose the MATLAB intention: data was (n_samples, n_features)
        X = X.T

    # Now X is (n_features, n_samples)
    n, m = X.shape
    
    if verbose:
        print(f"X.shape: {n}, {m}")
    
    if batch_size is None:
        batch_size = m

    sample_num = int(np.round(m * r))
    sample_num = max(1, min(sample_num, m))

    # Whitening on full X (like your MATLAB code)
    PCAWhite, U, S = pca_whiten_full(X, eps=eps)

    weights_concat = np.zeros((n, runs * n), dtype=np.float32)

    rng = np.random.default_rng(seed)

    for i in range(runs):
        idx = rng.integers(low=0, high=m, size=sample_num)  # with replacement
        xPCAWhite = PCAWhite[:, idx]  # (n_features, sample_num)

        # MATLAB rica(xPCAWhite', n): input is (sample_num, n_features), components=n
        Z = rica_fit_transformweights(
            xPCAWhite.T,               # (sample_num, n_features)
            n_components=n,            # match MATLAB
            lam=lam,
            sparsity=sparsity,
            lr=lr,
            batch_size=batch_size,
            epochs=epochs,
            seed=seed + i,
            device=device,
            verbose=verbose
        )

        # weight = u*diag(sqrt(diag(s)))*Mdl.TransformWeights;
        weight = (U @ np.diag(np.sqrt(S)) @ Z)  # (n_features, n_components)

        # weight = weight ./ repmat(max(abs(weight),[],1), n, 1);
        col_max = np.max(np.abs(weight), axis=0, keepdims=True)
        col_max[col_max == 0] = 1.0
        weight = weight / col_max
        
        weight = np.nan_to_num(weight, nan=0.0, posinf=0.0, neginf=0.0)

        weights_concat[:, i * n:(i + 1) * n] = weight

    weights_concat = np.nan_to_num(weights_concat, nan=0.0, posinf=0.0, neginf=0.0)
    support = identify_zero(weights_concat, l_num=n, thresh=thresh, alpha=alpha)
    return support, weights_concat
