# losses.py
# ==========================================================
import numpy as np


def mse_loss():
    """TODO: Compute mean squared reconstruction loss."""
    pass

def mae_loss():
    """TODO: Compute mean absolute reconstruction loss."""
    pass

def huber_loss():
    """TODO: Compute Huber loss."""
    pass

def gradient_loss():
    """TODO: Penalize gradient differences between spectra."""
    pass

def peak_loss():
    """TODO: Penalize errors around Raman peaks."""
    pass

def cosine_loss():
    """TODO: Penalize spectral angle differences."""
    pass

def combined_loss():
    """TODO: Combine multiple loss functions."""
    pass

def l1_regularization():
    """TODO: Compute L1 regularization."""
    pass

def l2_regularization():
    """TODO: Compute L2 regularization."""
    pass

def energy_loss(orig, processed):
    """
    Calculates the residual energy loss between an original (raw) spectrum and a processed spectrum. 
    It measures the total signal power lost during preprocessing relative to the original signal power.
    """
    orig_arr = np.asarray(orig)
    processed_arr = np.asarray(processed)
    
    residual_energy = np.sum((orig_arr - processed_arr) ** 2)
    orig_energy = np.sum(orig_arr ** 2)
    
    # Avoid division by zero
    if orig_energy == 0:
        return 0.0
        
    return residual_energy / orig_energy