import numpy as np
import math 

# Read-in
def read_csv(file_path, has_header=False, handle_missing=False):
    """
    Docstring
    """
    if handle_missing:
        # genfromtxt handles missing data, strings, and auto-detects data types
        data = np.genfromtxt(
            file_path, 
            delimiter=',', 
            skip_header=1 if has_header else 0,
            dtype=None,       # Automatically guess the column data types
            encoding='utf-8'  # Decodes bytes to strings cleanly
        )
    else:
        # loadtxt is faster but expects uniform, purely numeric tables
        data = np.loadtxt(
            file_path, 
            delimiter=',', 
            skiprows=1 if has_header else 0
        )
    return data

def read_txt(file_path):
    """
    Docstring
    """
    data = []
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                text = line
                spectra_loc = text.split(',')[0]
                amplitude = text.split(',')[1]
                data.append(i, spectra_loc, amplitude)
                i = i + 1
    except FileNotFoundError:
        print(f"Error: The file: {file_path} is not found.")

        structure = np.array(data, dtype=)
    return data

# Normalizations
def normalize(input):
    """
    son
    """
    normalized = np.deepcopy(input)
    
    return normalized


# Patch and Positional by kernel size 
def patch(structure, ker_size=15):
    """
    son
    """
    new_structure = np.zeroes()


    return new_structure



# Phase Location




