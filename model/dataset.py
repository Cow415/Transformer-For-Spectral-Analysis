import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split

class RamanDataset(Dataset):
    """
    PyTorch Dataset for paired Raman spectra.
    Each sample is expected to be a dictionary:
    {
        "shift": np.ndarray,
        "input": np.ndarray,
        "target": np.ndarray
    }
    """
    def __init__(
        self,
        pairs,
        transform=None
    ):
        self.pairs = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(
        self,
        idx
    ):

        sample = self.pairs[idx]

        x = sample["input"]
        y = sample["target"]

        if self.transform is not None:
            x = self.transform(x)

        x = torch.tensor(x, dtype=torch.float32)

        y = torch.tensor(y, dtype=torch.float32)

        # Conv1D expects (channels, length)
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)

        return x, y
    
def create_dataloaders(pairs, batch_size=32, train_ratio=0.8, val_ratio=0.1, shuffle=True, seed=42):
    """
    Create train/validation/test DataLoaders.

    Parameters
    ----------
    pairs : list
        List of paired spectra dictionaries.
    batch_size : int
    train_ratio : float
    val_ratio : float
    shuffle : bool
    seed : int

    Returns
    -------
    train_loader, val_loader, test_loader
    """
    dataset = RamanDataset(pairs)

    # Size each dataset group
    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size

    # Split data into 3 three groups
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=generator,)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle,)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,)

    return train_loader, val_loader, test_loader