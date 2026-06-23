"""Module holds functions for computation blocks to clean the notebook"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

import torch
import torch.nn as nn

from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import savgol_filter

# Corrections ==========================
# !!! BASELINE STABLIZER AND THE SG FILTERS AREN'T TESTED !!!
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
    diagonals = [np.ones(L), -2 * np.ones(L), np.ones(L)]
    D = sparse.spdiags(diagonals, [0, -1, -2], L, L-2)
    
    # Clculate penalty matrix: lam * (D * D^T)
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

def smooth_signal(data, window=11, order=2):
    """
    Smooths 1D data using a Savitzky-Golay filter.
    
    Parameters:
    - data: 1D array-like data to smooth.
    - window: Odd integer for sliding window size (default: 11).
    - order: Polynomial order, must be < window (default: 2).
    """
    # Quick guard rails to prevent SciPy crashes
    if window % 2 == 0:
        window += 1  # Force window to be odd
    if order >= window:
        order = window - 1  # Force polynomial order to be smaller
        
    # Apply filter
    smoothed = savgol_filter(data, window_length=window, polyorder=order)
    
    return smoothed

# Noiser ==========================
def gen_gauss_noise(clean, mu=0, sigma=0.02):
    noise = np.random.normal(mu, sigma*np.max(clean), clean.shape)
    noisy = clean + noise
    return noisy

def add_white_noise(spectrum, target_snr_db):
    # Convert input to a numpy array
    clean_signal = np.array(spectrum)
    
    signal_power = np.mean(clean_signal ** 2) # Average power of the clean signal
    # Convert target SNR from dB to a linear scale
    snr_linear = 10 ** (target_snr_db / 10)
    noise_power = signal_power / snr_linear     # Calculate the noise power to hit the target SNR
    
    # Calculate the standard deviation (sigma) of the noise
    noise_sigma = np.sqrt(noise_power)

    # Generate random Gaussian noise with the calculated standard deviation
    noise = np.random.normal(loc=0.0, scale=noise_sigma, size=clean_signal.shape)
    
    # Add the noise to the original clean signal
    noisy = clean_signal + noise

    return noisy

# Activation Functions ==========================
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def ReLu(x):
    return max(0, x)

def Soft_Max(x):
    e_x = np.exp(x - np.max(x)) # Subtract max for numerical stability
    return e_x / e_x.sum(axis=0)

def Leaky_ReLu(x, alpha=0.01):
    return x if x > 0 else alpha * x

# Loss Functions ==========================
def mse(sample, target):
    """
    Computes the Mean Squared Error (MSE) between two Raman spectra processed spectra against a reference library.
    """
    sample_arr = np.asarray(sample)
    target_arr = np.asarray(target)
    return np.mean((sample_arr - target_arr) ** 2)

def gradient_loss():
    return

def peak_region_loss():
    return

# Other Assessments ==========================
def mae(sample, target):
    """
    Computes the Mean Absolute Error (MAE) between two Raman spectra.
    """
    sample_arr = np.asarray(sample)
    target_arr = np.asarray(target)
    return np.mean(np.abs(sample_arr - target_arr))

# Backpropagation ==========================
# TODO
def compute_grad():

    return

def ADAM():

    return

def descend():

    return

