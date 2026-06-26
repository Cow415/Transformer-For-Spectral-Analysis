# visualization.py
# ==========================================================
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def plot_spectrum(shift, intensity, title=None):
    """
    Plot a single Raman spectrum.
    """
    # Plot
    plt.figure(figsize=(8, 4))
    plt.plot(shift, intensity)

    plt.xlabel("Raman Shift (cm$^{-1}$)")
    plt.ylabel("Intensity (a.u.)")
    if title:
        plt.title(title)

    plt.tight_layout
    plt.show()

def plot_overlay(shift, spectra, labels=None):
    """
    Overlay multiple spectra.
    """
    plt.figure(figsize=(8,4))

    for i,spectrum in enumerate(spectra):
        if labels:
            plt.plot(shift, spectrum, label=labels[i])
        else:
            plt.plot(shift, spectrum)
    if labels:
        plt.legend()

    plt.tight_layout()
    plt.show()

def plot_prediction(shift, input_spectrum, target, prediction):
    """
    Compare input, target, and prediction.
    """
    plt.figure(figsize=(8,4))
    
    plt.plot(shift, input_spectrum, label="Input")
    plt.plot(shift, target, label="Target")
    plt.plot(shift, prediction, label="Prediction")
    
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_residual(shift, target, prediction):
    """
    Plot residual spectrum.
    """
    residual = target - prediction

    plt.figure(figsize=(8,4))
    plt.plot(shift, residual)

    plt.title("Residual")
    plt.tight_layout
    plt.show()

def plot_peak_zoom(shift, intensity, lower, upper):
    """
    Zoom into selected Raman peaks.
    """
    mask = ((shift >= lower) & (shift <= upper))
    plot_spectrum(shift[mask], intensity[mask])

def plot_peak_comparison():
    """
    Compare detected peaks.
    """
    pass

def plot_training_curve(history):
    """
    Plot training and validation loss.
    """
    plt.figure(figsize=(6,4))
    plt.plot(history["train"], label="Train")

    plt.plot(history["val"], label="Validation")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_validation_curve(metric):
    """
    Plot validation metrics.
    """
    plt.figure(figsize=(6,4))
    plt.plot(metric)
    plt.tight_layout()
    plt.show()

def plot_latent_space_pca(latent ,labels=None):
    """
    Visualize latent space using PCA.
    """
    latent = PCA(n_components=2).fit_transform(latent)
    plt.figure(figsize=(6,6))

    if labels is None:
        plt.scatter(latent[:,0], latent[:,1])
    else:
        plt.scatter(latent[:,0], latent[:,1], c=labels)

    plt.tight_layout()
    plt.show()

def plot_latent_space_tsne(latent, labels=None):
    """
    Visualize latent space using t-SNE.
    """
    latent = TSNE(n_components=2).fit_transform(latent)
    plt.figure(figsize=(6,6))

    if labels is None:
        plt.scatter(latent[:,0], latent[:,1])
    else:
        plt.scatter(latent[:,0], latent[:,1], c=labels)

    plt.tight_layout()
    plt.show()

# ===
def plot_signature_decomposition(shift, input_spectrum, predicted_signature, reconstructed_background,):
    """
    Display the input spectrum, the estimated Raman signature,
    and the reconstructed background on three aligned subplots.
    """
    pass