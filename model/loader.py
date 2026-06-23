"""Module Provided Functions Process Raw Data"""
# Imports
import os
import math
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

# Normalizations
def normalize_vec(input_vector):
    """
    Reads a vector of values and normalized its amplitude out of 1
    
    Parameters:
    input (vec): input vector variable
    
    Returns:
    None
    """
    max_i = np.max(input_vector)
        # Handle the divide-by-zero case to avoid errors
    if max_i == 0:
        raise ValueError("Maximum amplitude is zero; cannot normalize.")

    # Perform division
    normalized = input_vector / max_i
    input_vector[:] = np.trunc(normalized * 1000000) / 1000000

    print(f"Intensity is normalized by dividing {max_i}.")

    return 

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

# Phrase Location
def phrase_locate(phrased_struct):
    """
    Identify the spectra window of each patch

    Parameters:
    phrased_struct (numpy.ndarray): vector structure indexed by position and holds its own nested vector
    ker_size (int): desired kernel size
    
    Returns:
    numpy.ndarray: New vector pair indexed with the start and end of each patch on the spectrum
    """
    struct = []
    i = 0
    for i in range(phrased_struct.shape[0]):
        start_pos = np.min(phrased_struct[i, 0, :])
        end_pos = np.max(phrased_struct[i, 0, :])
        struct.append([start_pos, end_pos])
        
    return np.array(struct)

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
