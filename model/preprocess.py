# preprocessing.py
# ==========================================================
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy import sparse
from scipy.sparse.linalg import spsolve

def baseline_als(intensity, lam=1e5, p=0.01, niter=10):
    """
    Asymmetric Least Squares baseline correction.

    Parameters:
        intensity (np.ndarray)
        lam (float): Smoothness penalty
        p (float): Asymmetry parameter
        niter (int): # iterate

    Returns:
        baseline (array): corrected spectrum
    """

    L = len(intensity)
    D = sparse.csc_matrix(np.diff(np.eye(L), 2))
                          
    w = np.ones(L)
    for _ in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.T)
        # Establish baseline
        baseline = spsolve(Z, w * intensity)
        w = np.where(intensity > baseline, p, 1-p)

    return baseline

def baseline_airpls():
    """TODO: Remove fluorescence using airPLS."""
    pass

def baseline_polynomial(shift, intensity, degree=5):
    """
    Remove baseline using polynomial fitting.
    
    Parameter: 
        shift (array):
        intensity (array):
        degree (int): polynomial degree
    
    Returns:
        new_baseline (array)
    """
    # Fit by polynomial and appy to baseline
    coeffs = np.polyfit(shift, intensity, degree)
    baseline = np.polyval(coeffs, shift)

    return baseline

def normalize_minmax(intensity):
    """
    Apply Min-Max normalization to intensity data.

    Parameters: 
        intensity (array): given data

    Returns: 
        Normalized array of same dimension
    """
    # Min-max
    minimum = np.min(intensity)
    maximum = np.max(intensity)
    # Shift by min and div by diff
    return (intensity - minimum) / (maximum - minimum + 1e-12)

def normalize_vector(intensity):
    """
    Normalize spectrum by vector norm.
    
    Parameters: 
        intensity (array): given data

    Returns: 
        Normalized array of same dimension
    """
    norm = np.linalg.norm(intensity)
    return intensity / (norm + 1e-12)

def normalize_area(shift, intensity):
    """
    Normalize spectrum by integrated area.
    
    Parameters: 
        shift (array): shift scale
        intensity (array): intensity data

    Returns: 
        Intensity array normalized by area
    """
    area = np.trapezoid(intensity, shift)
    return intensity / (area + 1e-12)

def normalize_zscore(intensity):
    """
    Apply z-score normalization to spectral intensity
    
    Parameters: 
        intensity (array)

    Returns:
        Normalized intensity
    """
    # Mean and std
    mean = np.mean(intensity)
    std = np.std(intensity)

    return (intensity - mean) / (std + 1e-12)

def crop_region(shift, intensity, lower, upper):
    """
    Crop spectrum to selected Raman shift range.
    
    Parameters:
        shift (array)
        intensity (array)
        lower (int): lower bound
        upper (int): upper bound
    
    Returns:
        Truncated shift and intensity
    """
    mask = ((shift >= lower) & (shift <= upper))
    
    return (shift[mask], intensity[mask])

def interpolate_resolution(shift, intensity, new_shift):
    """
    Interpolate spectra onto a common Raman axis.
    
    Parameters:
        shift (array)
        intensity (array)
        new_shift (array): New shift array w/ diff resolution to paste onto

    Returns:
        Truncated shift and intensity
    """
    interpolate = interp1d(shift, intensity, kind="linear", bounds_error=False, fill_value="extrapolate") # type: ignore
    return interpolate(new_shift)

def resample_spectrum(shift, intensity, num_pt=2097):
    """
    Resample spectrum to a fixed number of points.
    
    Parameters: 
        shift (array)
        intensity (array)
        num_pt (int): number of points on resampled spectrum

    Returns: 
        new sets of spectrum data
    """
    new_shift = np.linspace( shift.min(), shift.max(), num_pt)
    new_intensity = interpolate_resolution(shift, intensity, new_shift)

    return (new_shift, new_intensity)

def smooth_spectrum(intensity, window=11, order=3):
    """
    Apply optional smoothing filter(Savtik-Golay).
    
    Parameters:
        intensity (array)
        window (int): filter kernel size
        order (int): filter order
    
    Returns:
        Filtered intensity
    """
    # Quick guard rails to prevent SciPy crashes
    if window % 2 == 0:
        window += 1  # Force window to be odd
    if order >= window:
        order = window - 1  # Force polynomial order to be smaller
        
    return savgol_filter(intensity, window_length=window, polyorder=order)

def standardize_dataset(spectra, mode="vector"):
    """
    Standardize all spectra within the dataset.
    
    Parameters:
        spectra (array): list with all the dict
        mode (str): normalization method
    Returns: 
        Standardized spectra set
    """
    processed = []
    for spec in spectra:
        intensity = spec["intensity"]

        if mode == "vector":
            intensity = normalize_vector(intensity)
        elif mode == "minmax":
            intensity = normalize_minmax(intensity)

        # Append
        processed.append({**spec, "intensity": intensity})
    return processed

# === 
def preprocess_pair():
    pass