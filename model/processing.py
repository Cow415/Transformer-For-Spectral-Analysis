"""Module Provided Functions Used to Process Incoming Data"""
# Imports
import numpy as np

# Read-in
def read_csv(file_path, has_header=False, handle_missing=False):
    """
    Reads a csv file and extracts spectra location and amplitude.
    
    Parameters:
    file_path (str): Path to the target text file.
    has_header (bool): If True, skips the first line of the file(header).
    handle_missing (bool): if True, will need to handle inconsistent data

    Returns:
    numpy.ndarray: Structured or standard numpy array containing the data.
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
    return np.array(data)

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
            if with_head:
                next(file, None)           # Skip header
            for line in file:
                text = line.strip().split()     # Split by whitespace
                if len(text) >= 2:
                    spectra_loc = text[0]
                    amplitude = text[1]
                    data.append((spectra_loc, amplitude))
    except FileNotFoundError:
        print(f"Error: The file: {file_path} is not found.")

    return np.array(data)

# Normalizations
def normalize(input_vector):
    """
    Reads a vector structure of pair (spectra, amplitude), and normalized its amplitude out of 1
    
    Parameters:
    input (vec): input vector variable
    
    Returns:
    None
    """
    # Extract the amplitude array
    amplitude = np.array(input_vector[:, 1], dtype=float)
    max_i = np.max(amplitude)

    # Handle the divide-by-zero case to avoid errors
    if max_i == 0:
        raise ValueError("Maximum amplitude is zero; cannot normalize.")

    # Perform element-wise division
    normalized_amplitude = amplitude / max_i

    # Reconstruct the pair if you need to retain spectra
    input_vector[:, 1] = normalized_amplitude
    return

# Patch and Positional by kernel size
def patch(structure, ker_size=100):
    """
    Reads a vector structure of pair (spectra, amplitude)
    
    Parameters:
    file_path (str): Path to the target text file.
    ker_size (int): desired kernel size
    
    Returns:
    numpy.ndarray: New vector structure indexed by position and holds its own nested vector
    """
    vec_size = structure.shape[0]
    ker_num = int(vec_size / ker_size)

    patch_struct = []
    i = 0
    for i in range(ker_num):
        # Calculate boundaries for non-overlapping contiguous slices
        start_idx = i * ker_size
        end_idx = (i + 1) * ker_size

        patch_data = structure[start_idx:end_idx, :]
        patch_struct.append(patch_data)

    # Add truncated sets with zeropadding
    rem = vec_size % ker_size
    if rem > 0:
        truncated_set = structure[ker_num * ker_size:, :]
        # Calculate padding needed for axis 0
        pad_amount = ker_size - rem

        # Pad with zeros: 0 at the beginning, pad_amount at the end, 0 for the second axis
        padded_patch = np.pad(truncated_set, ((0, pad_amount), (0, 0)),
                              mode='constant', constant_values=0)
        patch_struct.append(padded_patch)

    return np.array(patch_struct)

# Phase Location
