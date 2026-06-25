# preprocessing.py
# ==========================================================
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy import sparse
from scipy.sparse.linalg import spsolve

def baseline_als():
    """TODO: Remove fluorescence using ALS baseline correction."""
    pass

def baseline_airpls():
    """TODO: Remove fluorescence using airPLS."""
    pass

def baseline_polynomial():
    """TODO: Remove baseline using polynomial fitting."""
    pass

def normalize_minmax():
    """TODO: Apply Min-Max normalization."""
    pass

def normalize_vector():
    """TODO: Normalize spectrum by vector norm."""
    pass

def normalize_area():
    """TODO: Normalize spectrum by integrated area."""
    pass

def normalize_zscore():
    """TODO: Apply z-score normalization."""
    pass

def crop_region():
    """TODO: Crop spectrum to selected Raman shift range."""
    pass

def interpolate_resolution():
    """TODO: Interpolate spectra onto a common Raman axis."""
    pass

def resample_spectrum():
    """TODO: Resample spectrum to a fixed number of points."""
    pass

def smooth_spectrum():
    """TODO: Apply optional smoothing filter."""
    pass

def standardize_dataset():
    """TODO: Standardize all spectra within the dataset."""
    pass
