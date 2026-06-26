# utils.py
# ==========================================================
import random 
import numpy as np 
import torch
import yaml

from torchinfo import summary

def set_seed(seed=42):
    """
    Set random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device():
    """
    Selects available CPU or GPU device.
    (NVIDIA, Silicon, or use CPU)
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    
    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")

def count_parameters(model):
    """
    Count trainable model parameters.
    """
    return sum(
        p.numel
        for p in model.parameters()
        if p.requires_grad
    )

def print_model_summary(model, input_size=(1,1,2097)):
    """
    Print network architecture summary.
    """
    summary(model, input_size=input_size)

def load_config(filepath):
    """
    Load project configuration.
    """
    with open(filepath) as f:
        config = yaml.safe_load(f)
    return config

def save_config(config, filepath):
    """
    Save project configuration.
    """
    with open(filepath,"w") as f:
            yaml.dump(config.f)