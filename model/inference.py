# inference.py
# ==========================================================
import pandas as pd
import numpy as np
import torch

@torch.no_grad()
def predict_single(model, spectrum, device):
    """
    Predict one Raman spectrum.
    """
    model.eval()
    x = torch.tensor(spectrum, dtype=torch.float32)
    x = x.to(device)
    prediction = model(x)
    return (prediction.cpu().squeeze.numpy())

@torch.no_grad()
def predict_batch(model, spectra, device):
    """
    Predict multiple Raman spectra.
    """
    outputs = []
    for spectrum in spectra: 
        outputs.append(predict_single(model, spectrum, device))
    return outputs 

# For residual structures
@torch.no_grad()
def remove_signature(model, spectrum, device):
    signature = predict_single(model, spectrum, device)
    return spectrum - signature

@torch.no_grad()
def extract_signature(model, spectrum, device):
    return predict_single(model, spectrum, device)

# Saving
def save_predictions(shift, prediction, filepath):
    df = pd.DataFrame({
        "Shift": shift,
        "Prediction": prediction
    })
    df.to_csv(filepath, index=False)
