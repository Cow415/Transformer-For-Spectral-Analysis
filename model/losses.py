# losses.py
# ==========================================================
# Assume output has the shape of (# in batch, spectra_size)
# 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def mse_loss(prediction, target):
    """ 
    Compute mean squared reconstruction loss.
    """
    return F.mse_loss(prediction, target)

def mae_loss(prediction, target):
    """
    Compute mean absolute reconstruction loss."""
    return F.l1_loss (prediction, target)

def huber_loss(prediction, target, delta=1.0):
    """
    Compute Huber loss.
    """
    return F.huber_loss(prediction, target, delta=delta)

# Raman Specified
def gradient_loss(prediction, target):
    """
    Penalize gradient differences between spectra.
    """
    # Compute gradient for each
    pred_grad = (prediction[..., 1:] - prediction[..., 1:])
    target_grad = (target[..., 1:] - target[..., 1:])
    return F.mse_loss(pred_grad, target_grad)

def peak_loss(prediction, target, peak_mask):
    """
    Penalize errors around Raman peaks.
    To make mask: peak_mask = ((shift >= __) & (shift <= __))
    """
    pred_peak = prediction[..., peak_mask]
    target_peak = prediction[..., peak_mask]
    return F.mse_loss(pred_peak, target_peak)

def cosine_loss(prediction, target):
    """
    Penalize spectral angle differences.
    """
    similarity = F.cosine_similarity(prediction, target, dim=-1)
    return (1 - similarity.mean())

def combined_loss(prediction, target, peak_mask=None):
    """
    Combine multiple loss functions.
    """
    # Weights can be tuned
    loss = (0.6 * mse_loss(prediction, target)
        + 0.2 * gradient_loss(prediction, target)
        + 0.2 * cosine_loss(prediction, target))
    if peak_mask is not None:
        loss += (0.1 * peak_loss(prediction, target, peak_mask))
    return loss

# Regularizations
def l1_regularization(model):
    """
    Compute L1 regularization.
    """
    reg = 0.0
    for param in model.parameter():
        reg += torch.sum(torch.abs(param))
    return reg

def l2_regularization(model):
    """TODO: Compute L2 regularization."""
    reg = 0.0 
    for param in model.parameters():
        reg += torch.sum(param**2)
    return reg

# Other
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