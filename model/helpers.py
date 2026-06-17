"""Module holds functions for computation blocks to clean the notebook"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

import torch
import torch.nn as nn

# Corrections ==========================
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


# !!! This is not revised !!!
def stabilize_raman_scipy(wavenumber_vector, intensity_vector, lam=1e6, p=0.01, max_iter=10):
    """
    Stabilizes and corrects a single Raman spectrum baseline using native SciPy.
    
    Parameters:
    -----------
    wavenumber_vector : X-axis, cm^-1
    intensity_vector : Y-axis, raw intensity
    lam (float): Smoothness parameter, default 1e6
    p (float): Asymmetry weighting parameter, default 0.01
    max_iter (int): Number of optimization iterations, default 10
    """
    # Ensure inputs are standard numpy arrays
    y = np.asarray(intensity_vector)
    L = len(y)
    
    # Construct the 2nd-order difference matrix (D) using sparse format
    # This prevents memory explosion on long spectral lengths
    diagonals = [np.ones(L), -2 * np.ones(L), np.ones(L)]
    D = sparse.spdiags(diagonals, [0, -1, -2], L, L-2)
    
    # Pre-calculate penalty matrix: lam * (D * D^T)
    DD_t = lam * D.dot(D.transpose())
    
    # Initialize baseline weights uniformly
    w = np.ones(L)
    z = np.zeros(L)
    
    # Iteratively solve for the underlying baseline
    for _ in range(max_iter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + DD_t
        z = spsolve(Z, w * y)
        
        # Penalize points above the baseline (p) and reward points below it (1-p)
        w = p * (y > z) + (1 - p) * (y < z)
        
    # Subtract baseline to stabilize intensities at zero
    corrected_intensity = y - z
    return corrected_intensity

# Noiser ==========================
def gen_noise(clean):
    noise = np.random.normal(0, 0.02*np.max(clean), clean.shape)
    noisy = clean + noise
    return noisy

# Activation Functions ==========================


# Loss Functions ==========================


# Backpropagation ==========================