# benchmark.py
# ==========================================================
import numpy as np
import pandas as pd
import pywt
import torch
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA

from inference import (predict_single, remove_signature)
from metrics import (
    compute_rmse,
    compute_mae,
    compute_cosine_similarity,
    compute_pearson,
    compute_snr,
)

# Popular Processors
def run_savgol(spectrum, window=11, poly=3):
    """
    Run Savitzky-Golay baseline method.
    """
    return savgol_filter(spectrum, window, poly)

def run_wavelet(spectrum, wavelet="db4", level=3, threshold_scale=0.5):
    """
    Run wavelet denoising baseline.
    """
    coeffs = pywt.wavedec(spectrum, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = (threshold_scale * sigma * np.sqrt(2 * np.log(len(spectrum))))

    denoised_coeffs = [coeffs[0]]
    for detail in coeffs[1:]:
        denoised_coeffs.append(pywt.threshold(detail, threshold, mode="soft"))

    reconstruction = pywt.waverec(denoised_coeffs, wavelet)
    return reconstruction[:len(spectrum)]

def run_gaussian_filter(spectrum, sigma=2):
    """
    Run Gaussian smoothing baseline.
    """
    return gaussian_filter1d(spectrum, sigma)

def run_pca_denoising(spectra, n_comp=10):
    """
    Run PCA reconstruction baseline.
    """
    pca = PCA(n_components=n_comp)
    latent = pca.fit_transform(spectra)
    reconstruction = pca.inverse_transform(latent)
    return reconstruction

# Model
def run_autoencoder(model, spectrum, device):
    """
    Run trained autoencoder.
    """
    return predict_single(model, spectrum, device)

def run_residual_autoencoder(model, spectrum, device):
    """
    Run residual autoencoder.
    """
    return remove_signature(model, spectrum, device)

def benchmark_models(methods, inputs, targets):
    """
    Evaluate all methods on identical data.
    """
    result = {}

    for name, method in methods.items():
        metrics = []
        for x, y in zip(inputs, targets):
            pred = method(x)
            metrics.append(compare_metrics(pred, y))
        result[name] = metrics
    return result

# Comparison
def compare_metrics(prediction, target):
    """
    Compare evaluation metrics across methods.
    """
    return {
        "RMSE":
            compute_rmse(prediction, target),
        "MAE":
            compute_mae(prediction, target),
        "COSINE":
            compute_cosine_similarity(prediction, target),
        "Pearson": 
            compute_pearson(prediction, target),
        "SNR":
            compute_snr(prediction, target)
    }

def generate_summary_table(results):
    """
    Generate benchmark results table.
    """
    rows = []
    for method, metrics in results.items():
        row = {
            "Method": method,

            "RMSE": float(np.mean([m["RMSE"] for m in metrics])),

            "MAE": float(np.mean([m["MAE"] for m in metrics])),

            "COSINE": float(np.mean([m["COSINE"] for m in metrics])),

            "PEARSON": float(np.mean([m["PEARSON"] for m in metrics])),

            "SNR": float(np.mean([m["PEARSON"] for m in metrics]))
        }
        rows.append(row)
    return pd.DataFrame(rows)
