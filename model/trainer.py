# trainer.py
# ==========================================================
import numpy as np
import pandas as pd
import torch
from losses import combined_loss
from torch.optim.lr_scheduler import ReduceLROnPlateau

def train_epoch(model, loader, optimizer, device):
    """
    Train the model for one epoch.
    """
    model.train()
    running_loss = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()

        prediction = model(x)
        loss = combined_loss(prediction, y)

        loss.backward()
        optimizer.step()

        running_loss += (loss.item())
    return (running_loss / len(loader))

@torch.no_grad()
def validate_epoch(model, loader, device):
    """
    Evaluate the model on validation data.
    """
    model.eval()
    running_loss = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        prediction = model(x)
        loss = combined_loss(prediction, y)

        running_loss += (loss.item())
    return (running_loss / len(loader))
    
def save_checkpoint(model, optimizer, epoch, filepath):
    """
    Save model checkpoint.
    """
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict()
        }, filepath)

def load_checkpoint(model, optimizer, filepath):
    """
    Load model checkpoint.
    """
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["epoch"]

def train_model(model, train_loader, val_loader, optimizer, epochs, device):
    """
    Execute the complete training pipeline.
    """
    history = {"train": [], "val": []}

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = validate_epoch(model, val_loader, device)

        history["train"].append(train_loss)
        history["val"].append(val_loss)

        print(
            f"Epoch {epoch+1}/{epochs}"
            f" | Train={train_loss:.6f}"
            f" | Val={val_loss:.6f}" )
    return history

# Other Training Functions
class EarlyStopping:
    """
    Determine whether to stop training depending on validation loss.
    
    Usage:
    early_stop = EarlyStopping(patience=15)
    ...
    if early_stop(val_loss):
        break
    """
    def __init__(self, patience=20, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = np.inf
        self.counter = 0

    def __call__(self, validation_loss):
        if validation_loss < self.best_loss - self.min_delta:
            self.best_loss = validation_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience

def adjust_learning_rate(optimizer, validation_loss):
    """
    Update optimizer learning rate using scheduler interface, build a scheduler to use.
    !!! May need to sparately define scheduler to use !!!
    """
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    scheduler.step(validation_loss)

def log_metrics(history, train_loss, val_loss):
    """
    Record training metrics.
    """
    history["train"].append(train_loss)
    history["val"].append(val_loss)

def save_training_history(history, filepath):
    """
    Save loss history.
    """
    pd.DataFrame(history).to_csv(filepath, index=False)
