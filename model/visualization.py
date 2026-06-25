# visualization.py
# ==========================================================
import numpy as np
from matplotlib import pyplot as plt


def plot_spectrum(spectrum):
    """TODO: Plot a single Raman spectrum."""
    # Graph
    x = spectrum["shift"]
    y = spectrum["intensity"]

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(x, y)

    plt.yticks([])

    plt.title("Raman Spectrum")
    plt.xlabel("Raman Shift (cm$^{-1}$)")
    plt.ylabel("Intensity (a.u.)")
    plt.show()
    pass

def plot_overlay():
    """TODO: Overlay multiple spectra."""
    pass

def plot_prediction():
    """TODO: Compare input, target, and prediction."""
    pass

def plot_residual():
    """TODO: Plot residual spectrum."""
    pass

def plot_peak_zoom():
    """TODO: Zoom into selected Raman peaks."""
    pass

def plot_peak_comparison():
    """TODO: Compare detected peaks."""
    pass

def plot_training_curve():
    """TODO: Plot training and validation loss."""
    pass

def plot_validation_curve():
    """TODO: Plot validation metrics."""
    pass

def plot_latent_space_pca():
    """TODO: Visualize latent space using PCA."""
    pass

def plot_latent_space_tsne():
    """TODO: Visualize latent space using t-SNE."""
    pass
