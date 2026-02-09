# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 16:12:16 2026

@author: chdem
"""

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from scipy.stats import ttest_1samp


# -----------------------------
# Whitening (as in the paper)
# -----------------------------
def whiten(X, eps=1e-12):
    """
    X: (n_samples, n_features)
    Returns:
      W: whitened data (n_samples, n_features)
      mean: (n_features,)
      U: (n_features, n_features) eigenvectors
      S: (n_features,) eigenvalues
    """
    X = np.asarray(X, dtype=np.float64)
    mean = X.mean(axis=0, keepdims=True)
    Xc = X - mean

    # covariance
    C = (Xc.T @ Xc) / max(1, (Xc.shape[0] - 1))

    # eigh for symmetric matrices
    S, U = np.linalg.eigh(C)  # S ascending
    idx = np.argsort(S)[::-1]  # descending
    S = S[idx]
    U = U[:, idx]

    # whitening transform: Xw = (X - mean) @ U @ diag(1/sqrt(S))
    inv_sqrt = 1.0 / np.sqrt(S + eps)
    W = Xc @ U @ np.diag(inv_sqrt)

    return W.astype(np.float32), mean.squeeze(0).astype(np.float32), U.astype(np.float32), S.astype(np.float32)


# -----------------------------
# RICA model
# -----------------------------
class RICA(nn.Module):
    """
    RICA with tied weights:
      h = Z^T w
      w_hat = Z h = Z Z^T w

    We learn Z (shape: n_features x n_components).
    """
    def __init__(self, n_features, n_components):
        super().__init__()
        # Z: (n_features, n_components)
        Z = torch.randn(n_features, n_components) * 0.01
        self.Z = nn.Parameter(Z)

    def forward(self, W):
        """
        W: (batch, n_features)
        Returns:
          H: (batch, n_components)
          W_hat: (batch, n_features)
        """
        H = W @ self.Z          # (batch, n_components)  since H = Z^T w, w row-vector => w @ Z
        W_hat = H @ self.Z.T    # (batch, n_features)    reconstruction
        return H, W_hat


def rica_train(
    X,
    n_components,
    lam=0.1,
    sparsity="logcosh",   # "logcosh" or "l1"
    lr=1e-2,
    batch_size=512,
    epochs=2000,
    device=None,
    seed=0,
    verbose=200
):
    """
    Trains RICA on X and returns:
      B_tilde_prime (mixing matrix estimate in the original space)
      Z_star (in whitened space)
      whitening params (mean, U, S)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # 1) whiten
    W_np, mean, U, S = whiten(X)
    W = torch.from_numpy(W_np).to(device)

    n_samples, n_features = W.shape
    model = RICA(n_features, n_components).to(device)
    opt = Adam(model.parameters(), lr=lr)

    def sparsity_penalty(H):
        # H is (batch, n_components)
        if sparsity == "l1":
            return H.abs().mean()
        elif sparsity == "logcosh":
            # smooth sparsity proxy; stable for training
            return torch.log(torch.cosh(H)).mean()
        else:
            raise ValueError("sparsity must be 'logcosh' or 'l1'")

    # mini-batch training
    for ep in range(1, epochs + 1):
        # shuffle indices
        idx = rng.permutation(n_samples)
        total_loss = 0.0

        for start in range(0, n_samples, batch_size):
            bidx = idx[start:start + batch_size]
            Wb = W[bidx]

            H, W_hat = model(Wb)

            sp = sparsity_penalty(H)
            rec = ((W_hat - Wb) ** 2).mean()  # MSE reconstruction penalty

            loss = sp + lam * rec

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item() * len(bidx)

        if verbose and (ep % verbose == 0 or ep == 1 or ep == epochs):
            print(f"epoch {ep:5d} | loss {total_loss / n_samples:.6f}")

    # 2) Recover B_tilde' in original (unwhitened) observed space:
    # Paper uses: B_tilde'' ≈ U Σ^{1/2} Z*
    # Here Σ is eigenvalues; Σ^{1/2} is diag(sqrt(S))
    Z_star = model.Z.detach().cpu().numpy()  # (n_features, n_components)
    B_tilde_prime = (U @ np.diag(np.sqrt(S)) @ Z_star).astype(np.float32)

    return B_tilde_prime, Z_star, (mean, U, S)







def bootstrap_rica(
    X,
    n_components,
    r=20,
    alpha=0.05,
    seed=0,
    **rica_kwargs
):
    """
    Parameters
    ----------
    X : ndarray (n_samples, p_observed)
    n_components : int
    r : number of bootstrap resamples
    alpha : significance level for zero testing
    rica_kwargs : forwarded to fit_rica_get_Btilde_prime

    Returns
    -------
    B_mean : mean B_tilde_prime across runs
    B_std : std deviation across runs
    support : boolean mask where entry is significantly non-zero
    """

    rng = np.random.default_rng(seed)
    n_samples = X.shape[0]

    B_list = []

    for i in range(r):
        # resample with replacement
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        X_boot = X[indices]

        B, _, _ = rica_train(
            X_boot,
            n_components=n_components,
            **rica_kwargs
        )

        B_list.append(B)

    B_stack = np.stack(B_list, axis=0)  # shape (r, p, q)

    # Align columns by absolute max (simple heuristic)
    # Normalize each column so max abs entry = 1
    for i in range(r):
        for j in range(B_stack.shape[2]):
            col = B_stack[i, :, j]
            max_idx = np.argmax(np.abs(col))
            if col[max_idx] != 0:
                B_stack[i, :, j] /= col[max_idx]

    # Mean & std
    B_mean = B_stack.mean(axis=0)
    B_std = B_stack.std(axis=0)

    # Statistical test: is entry significantly different from zero?
    tvals, pvals = ttest_1samp(B_stack, popmean=0.0, axis=0)
    support = pvals < alpha

    return B_mean, B_std, support

# -----------------------------
# Example usage
# -----------------------------

"""
if __name__ == "__main__":
    # X: (n_samples, p_o) observed data
    n = 5000
    p = 5
    X = np.random.randn(n, p).astype(np.float32)

    # overcomplete: components > observed dims
    B_tilde_prime, Z_star, (mean, U, S) = rica_train(
        X,
        n_components=10,
        lam=0.5,
        sparsity="logcosh",
        lr=1e-2,
        batch_size=512,
        epochs=2000,
        verbose=200
    )

    print("B_tilde_prime shape:", B_tilde_prime.shape)  # (p_o, n_components)
"""