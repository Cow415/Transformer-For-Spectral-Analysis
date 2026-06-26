# pipeline.py
# ==========================================================
import os
import numpy as np 
import torch
from torch.utils.data import DataLoader

from loader import *
from dataset import *
from preprocess import *
from trainer import *
from ref_structure import *
from benchmark import *
from inference import *
from utils import *

def preprocess_pipeline(config):
    """
    Load folders
      ↓
    Pair datasets
      ↓
    Validate pairs
      ↓
    Baseline correction
      ↓
    Normalization
      ↓
    Optional augmentation
      ↓
    Return processed pairs
    """
     # Load spectra
    input_spectra = load_folder(config["input_folder"])
    target_spectra = load_folder(config["target_folder"])

    # Pair samples
    pairs = pair_datasets(
        config["input_folder"],
        config["target_folder"]
    )

    processed_pairs = []

    for pair in pairs:

        input_spec = pair["input"]
        target_spec = pair["target"]

        # Baseline correction
        input_baseline = baseline_als(input_spec["intensity"])
        target_baseline = baseline_als(target_spec["intensity"])

        input_corrected = input_spec["intensity"] - input_baseline
        target_corrected = target_spec["intensity"] - target_baseline

        # Normalization
        input_corrected = normalize_vector(input_corrected)
        target_corrected = normalize_vector(target_corrected)

        processed_pairs.append({
            "shift": input_spec["shift"],
            "input": input_corrected,
            "target": target_corrected
        })
    return processed_pairs

def train_pipeline(config):
    """
    Processed pairs
        ↓
    Dataset
        ↓
    DataLoader
        ↓
    Create model
        ↓
    Optimizer
        ↓
    Scheduler
        ↓
    Train
        ↓
    Save checkpoint
    """
    pairs = preprocess_pipeline(config)

    train_dataset = RamanDataset(pairs)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)

    device = get_device()
    model = ResidualAutoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    history = train_model(
        model,
        train_loader,
        train_loader,          # replace with validation loader
        optimizer,
        config["epochs"],
        device
    )

    save_checkpoint(
        model,
        optimizer,
        config["epochs"],
        config["checkpoint"]
    )

    return model, history

def evaluation_pipeline(model, test_loader, device):
    """Load model
      ↓
    Predict
      ↓
    Compute metrics
      ↓
    Visualize
      ↓
    Return report
    """
    metrics = []

    for x, y in test_loader:
        prediction = model(x.to(device))
        prediction = prediction.cpu().numpy()

        y = y.numpy()
        metrics.append(compare_metrics(prediction.squeeze(), y.squeeze()))

    return generate_summary_table({"Autoencoder": metrics})

def benchmark_pipeline(spectra, targets, model, device):
    """
    Spectrum
    ↓
    Savitzky-Golay
    Wavelet
    Gaussian
    PCA
    Autoencoder
    Residual AE
    ↓
    Metrics
    ↓
    Summary table
    """
    methods = {
        "SavGol":
            lambda x: run_savgol(x),
        "Wavelet":
            lambda x: run_wavelet(x),
        "Gaussian":
            lambda x: run_gaussian_filter(x),
        "Autoencoder":
            lambda x: predict_single(model, x, device)
    }

    results = benchmark_models(methods, spectra, targets)
    return generate_summary_table(results)

def inference_pipeline(model, filepath, device):
    """New Raman spectrum
        ↓
    Preprocessing
        ↓
    Model prediction
        ↓
    Background
        +
    Signature
        ↓
    Save result
    """
    spectrum = load_csv(filepath)
    baseline = baseline_als(spectrum["intensity"])
    corrected = (spectrum["intensity"] - baseline)

    corrected = normalize_vector(corrected)
    prediction = predict_single(model, corrected, device)
    return prediction

#===
def decomposition_pipeline(model, filepath, device):
    """
    Process a single spectrum and return:
      - Original spectrum
      - Estimated Raman signature
      - Reconstructed background
      - Evaluation plots
      - Saved CSV outputs
    """
    pass