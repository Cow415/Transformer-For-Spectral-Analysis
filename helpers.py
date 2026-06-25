"""Module holds functions for computation blocks to clean the notebook"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

import torch
import torch.nn as nn

from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import savgol_filter

# Noiser ==========================
def add_white_noise(spectrum, target_snr_db):
    # Convert input to a numpy array
    clean_signal = np.array(spectrum)
    
    signal_power = np.mean(clean_signal ** 2) # Average power of the clean signal
    # Convert target SNR from dB to a linear scale
    snr_linear = 10 ** (target_snr_db / 10)
    noise_power = signal_power / snr_linear     # Calculate the noise power to hit the target SNR
    
    # Calculate the standard deviation (sigma) of the noise
    noise_sigma = np.sqrt(noise_power)

    # Generate random Gaussian noise with the calculated standard deviation
    noise = np.random.normal(loc=0.0, scale=noise_sigma, size=clean_signal.shape)
    
    # Add the noise to the original clean signal
    noisy = clean_signal + noise

    return noisy

# Activation Functions ==========================
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def ReLu(x):
    return max(0, x)

def Soft_Max(x):
    e_x = np.exp(x - np.max(x)) # Subtract max for numerical stability
    return e_x / e_x.sum(axis=0)

def Leaky_ReLu(x, alpha=0.01):
    return x if x > 0 else alpha * x

"""Module Provided Functions Load Raw Data"""
# Imports
import os
import math
import numpy as np

# Read-in .txt files
def read_txt(file_path, with_head=False):
    """
    Reads a space-separated text file and extracts spectra location and amplitude.
    
    Parameters:
    file_path (str): Path to the target text file.
    with_head (bool): If True, skips the first line of the file(header).
    
    Returns:
    numpy.ndarray: Structured or standard numpy array containing the data.
    """
    data = []
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            spectra_loc = []
            amplitude = []
            if with_head:
                next(file, None)           # Skip header
            for line in file:
                text = line.strip().split()     # Split by whitespace
                if len(text) >= 2:
                    spectra_loc.append(text[0])
                    amplitude.append(text[1])
            data.append(spectra_loc)
            data.append(amplitude)
    except FileNotFoundError:
        print(f"Error: The file: {file_path} is not found.")

    return np.array(data, dtype=float)

# Patch and Positional by kernel size
def patch(structure, ker_size=100):
    """
    Reads n-vectors structure, break into tokenizable patches of data
    
    Parameters:
    structure (numpy.ndarray): The target structural data matrix (usually 2D, e.g., shape: [sets, sequence_length]).
    ker_size (int): desired kernel size
    
    Returns:
    numpy.ndarray: New vector structure indexed by position and holds its own nested vector
    """
    vec_size = structure.shape[1]
    ker_num = int(vec_size / ker_size)
    patch_struct = []

    i = 0
    for i in range(ker_num):
        # Calculate boundaries for non-overlapping contiguous slices
        start_idx = i * ker_size
        end_idx = (i + 1) * ker_size

        patch_data = structure[:, start_idx:end_idx]
        patch_struct.append(patch_data)

    # Add truncated sets with zeropadding
    rem = vec_size % ker_size
    if rem > 0:
        truncated_set = structure[:, ker_num * ker_size:]
        # Calculate padding needed for axis 0
        pad_amount = ker_size - rem

        # Pad with zeros: 0 at the beginning, pad_amount at the end, 0 for the second axis
        padded_patch = np.pad(truncated_set, ((0, 0), (0, pad_amount)), mode='constant', constant_values=0)
        patch_struct.append(padded_patch)

    return np.array(patch_struct)

# ===========================================================
# Dataset Splitting and Batches
def read_txt_bulk(list_path, paths_wd, with_head=False, inten_col_idx=1):
    """
    Reads a text file with a list paths for the given data_set and extracts spectra intensity from each file.

    !!! Assume all the files accessed in the list have the same spectra sampling locations
    
    Parameters:
    path_list (str): Path to the target text file with all the directories.
    paths_wd (str): Expected folder for the set of files
    with_head (bool): If True, skips the first line of the file(header).
    inten_col (int): Column for the intensity data
    
    Returns:
    numpy.ndarray: Structured or standard numpy array containing the data.
    """
    # Set-up directory
    print(f"Looking into {paths_wd}") 
    if not os.path.isabs(list_path) and not os.path.exists(list_path):
        list_path = os.path.join(paths_wd, list_path)

    # Read in each file path
    paths = []
    try:
        with open(list_path, "r", encoding="utf-8") as file:
            for line in file:
                path = line.strip()         # Strip newline 
                if path:
                    if not os.path.isabs(path):
                        path = os.path.join(paths_wd, path)
                    paths.append(path)
    except FileNotFoundError:
        print(f"Error: The file: {list_path} is not found.")
        return np.array([])

    if not paths:
        print(f"Error: The file: {list_path} contains no file paths.")
        return np.array([])

    # Run down extract first set
    premiere = paths.pop(0)
    print(f"Accessing Set[1]: {premiere}")
    data_set = read_txt(premiere, with_head)
    
    set_num = 2
    for path in paths: 
        spectra = []

        try:
            print(f"Accessing Set[{set_num}]: {path}")
            with open(path, "r", encoding="utf-8") as file:
                if with_head:
                    next(file, None)           # Skip header
                for line in file:
                    text = line.strip().split()     # Split by whitespace
                    spectra.append(text[inten_col_idx])     # Read each row in that column

            try:
                data_set = np.vstack([data_set, np.array(spectra, dtype=float)])        # Append the spectra data over
            except ValueError: 
                print(f"Error in set{set_num}: {path}")
        except FileNotFoundError:
            print(f"Error: The file: {path} is not found.")
        set_num = set_num + 1

    data_set = np.array(data_set, dtype=float)
    print(f"Number of dataset: {data_set.shape[0] - 1}")
    return data_set


"""Module Holds the Architecture Component Templates"""
# Data structure node formats to use
import numpy as np
import torch
import torch.nn as nn

# Patch Embeddings
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=96, patch_size=16, num_hiddens=512):
        super().__init__()
        def _make_tuple(x):
            if not isinstance(x, (list, tuple)):
                return (x, x)
            return x
        img_size, patch_size = _make_tuple(img_size), _make_tuple(patch_size)
        self.num_patches = (img_size[0] // patch_size[0]) * (
            img_size[1] // patch_size[1])
        self.conv = nn.LazyConv2d(num_hiddens, kernel_size=patch_size,
                                  stride=patch_size)

    def forward(self, X):
        # Output shape: (batch size, no. of patches, no. of channels)
        return self.conv(X).flatten(2).transpose(1, 2)
    
# =======================================================================
# Denoising Autoencoder
# Class Encoder
class Encoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(1,16,7,padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(16,32,7,padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32,64,5,padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

    def forward(self,x):
        return self.net(x)

# Class Decoder
class Sig_Decoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.ConvTranspose1d(
                64,32,
                kernel_size=2,
                stride=2
            ),
            nn.ReLU(),

            nn.ConvTranspose1d(
                32,16,
                kernel_size=2,
                stride=2
            ),
            nn.ReLU(),

            nn.ConvTranspose1d(
                16,1,
                kernel_size=2,
                stride=2
            )
        )

    def forward(self,z):
        return self.net(z)
    
class BG_Decoder(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.ConvTranspose1d(
                64,32,
                kernel_size=2,
                stride=2
            ),
            nn.ReLU(),

            nn.ConvTranspose1d(
                32,16,
                kernel_size=2,
                stride=2
            ),
            nn.ReLU(),

            nn.ConvTranspose1d(
                16,1,
                kernel_size=2,
                stride=2
            )
        )

    def forward(self,z):
        return self.net(z)        
