# loader.py
# ==========================================================
import os
from pathlib import Path
import numpy as np
import pandas as pd

# Loading and Saving
def load_csv(filepath):
    """
    Load a Raman spectrum from a CSV file.
    
    Parameters: 
        filepath (string): String of the access path to particular file

    Returns:
    dict {
        "filename": str,
        "shift": np.ndarray,
        "intensity": np.ndarray
         }
    """
    df = pd.read_csv(filepath)
    shift = df.iloc[:, 0].to_numpy(dtype=np.float32)
    intensity = df.iloc[:, 1].to_numpy(dtype=np.float32)

    return {
        "filename": Path(filepath).name,
        "shift": shift,
        "intensity": intensity
    }

def load_txt(filepath):
    """
    Load a Raman spectrum from a TXT file.
    
    Parameters: 
        filepath (string): String of the access path to particular file

    Returns:
    dict {
        "filename": str,
        "shift": np.ndarray,
        "intensity": np.ndarray
         }
    """
    df = pd.read_csv(filepath, sep=r"\s+", header=None)
    shift = df.iloc[:, 0].to_numpy(dtype=np.float32)
    intensity = df.iloc[:, 1].to_numpy(dtype=np.float32)

    return {
        "filename": Path(filepath).name,
        "shift": shift,
        "intensity": intensity
    }

def load_excel(filepath):
    """
    Load a Raman spectrum from a Excel file.
    
    Parameters: 
        filepath (string): String of the access path to particular file

    Returns:
    dict {
        "filename": str,
        "shift": np.ndarray,
        "intensity": np.ndarray
        }
    """
    df = pd.read_excel(filepath)
    shift = df.iloc[:,0].to_numpy(dtype=np.float32)
    intensity = df.iloc[:,1].to_numpy(dtype=np.float32)

    return {
        "filename": Path(filepath).name,
        "shift": shift,
        "intensity": intensity
    }

def load_folder(folder):
    """
    Load every file in the folder(csv, txt, or xlsx).
    To use: "spectra = load_folder("data/)"
    
    Parameters: 
        filepath (string): String of the access path to particular file

    Returns:
        spectra[] (dict array): Array of dict for all loaded data
    """

    folder = Path(folder)
    spectra = []

    for file in sorted(folder.iterdir()):

        suffix = file.suffix.lower()

        # Load according to each file format
        if suffix == ".csv":
            spectra.append(load_csv(file))
        elif suffix == ".txt":
            spectra.append(load_txt(file))
        elif suffix in [".xlsx", ".xls"]:
            spectra.append(load_excel(file))

    return spectra  # Return array with all the loaded dict

def save_csv(data, filepath):
    """
    Save a given spectra data as a CSV file.
    
    Parameters: 
        data (dict): Format of output data with filename, shift, and intensity
        filepath (string): String pertaining the location to save the spectra

    Returns: None
    """
    df = pd.DataFrame({     # Take only the spectra and shift
        "Shift": data["shift"], 
        "Intensity": data["shift"]})
    df.to_csv(filepath, index=False)    # Save

def save_model_output(shift, prediction, filepath):
    """
    Save an output spectra data from the model as a CSV file.
    
    Parameters: 
        shift (array): given shift axis
        prediction (array): predicted intensity
        filepath (string): String holding the file path to save this as
    
    Returns: None
    """
    df = pd.DataFrame({     # Build saved dataset
        "Shift": shift,
        "Prediction": prediction
    })
    df.to_csv(filepath, index=False)    # Save

# Input-Output Pairing
def match_sample_names(input_folder, target_folder):
    """
    Checks the filenames in each folder that matches

    Parameters:
        input_folder (string): path for input folder
        output_folder (string):  path for output folder
    Returns:
        matches(array): list for every matching files in pairs
    """
    # Assign paths var
    input_folder = Path(input_folder)
    target_folder = Path(target_folder)
    matches = []
    # Loop to check for corresponding names
    for input_file in input_folder.glob("*"):
        # Pull file in output for the same name
        target_file = target_folder / input_file.name

        if target_file.exists():
            matches.append((input_file, target_file))

    return matches

def pair_datasets(input_folder, target_folder):
    """
    Pair spectra from both dataset to exam
    
    Parameters:
        input_folder (string): path for input folder
        output_folder (string):  path for output folder
    Returns:
        pairs(array): list for matching datas in pairs as dict
    """
    pairs = []

    matches = match_sample_names(input_folder, target_folder)

    # Load and save the pairs
    for input_file, target_file in matches:
        
        input_spec = load_csv(input_file)
        target_spec = load_csv(target_file)

        pairs.append({
            "input": input_spec,
            "target": target_spec
        })
    return pairs

# Validations
def validate_lengths(spectra, expected_len):
    """
    Ensure every spectrum has is in the desirable length.
    
    Parameters: 
        spectra (array): array of lists of spectrum datas
        expected_len (int): Size the length should be

    Return
        invalid (array): an array of all spectra with differing spectrum size.
    """
    invalid = []
    for idx, spectrum in enumerate(spectra):
        n = len(spectrum["intensity"])  # Load spectra
        if n != expected_len:
            invalid.append(idx)
    return invalid    

def validate_shift_axis(spectrum_a, spectrum_b, atol=1e-6):
    """
    Verify Raman shift axes are identical.

    Parameters: 
        spectrum_a (dict): Data for one of the spectrum
        spectrum_b (dict): Data for another
        atol (float): tolerance of the difference

    Returns:
        bool: if two arrayas are element-wise approx. equal
    """
    return np.allclose(spectrum_a["shift"], spectrum_b["shift"], atol=atol)

def validate_pair_axis(pairs, tolerance=1e-6):
    """
    Check if all pair's axis are identical
    
    Parameter:
        pair (array): data pairs of dict
        tolerance (flaot): tolerance of difference

    Returns:
        bad_pair (array): pairs that failed the validation
    """
    bad_pairs = []

    for idx, pair in enumerate(pairs):

        valid = validate_shift_axis(pair["input"], pair["target"], tolerance)
        if not valid:
            bad_pairs.append(idx)

    return bad_pairs

def check_missing_values(spectrum):
    """
    Detect NaN or invalid intensity values.

    Parameters:
        spectrum (dict): dataset for a particular spectrum

    Returns:
        true if bad value
    """
    intensity = spectrum["intensity"]

    has_nan = np.isnan(intensity).any()
    has_inf = np.isinf(intensity).any()
    return has_nan or has_inf

def remove_invalid_samples(spectra):
    """TODO: Remove corrupted spectra from the given dataset."""
    clean = []

    for spectrum in spectra:
        # Check if data is clean
        if not check_missing_values(spectrum):
            clean.append(spectrum)

    return clean
