# metrics.py
# ==========================================================
import numpy as np

def compute_mse(prediction, target):
    """
    Compute Mean Squared Error.
    """
    prediction = np.asarray(prediction)
    target = np.asarray(target)
    return np.mean((prediction - target)**2)

def compute_rmse(prediction, target):
    """
    Compute Root Mean Squared Error.
    """
    return np.sqrt(compute_mse(prediction, target))

def compute_mae(prediction, target):
    """
    Compute Mean Absolute Error.
    """
    prediction = np.asarray(prediction)
    target = np.asarray(target)
    return np.mean(np.abs(prediction - target))

def compute_r2(prediction, target):
    """
    Compute R² score.
    """
    prediction = np.asarray(prediction)
    target = np.asarray(target)

    ss_res = np.sum((target - prediction)**2)
    ss_tot = np.sum(target - np.mean(target)**2)
    return 1 - (ss_res/(ss_tot + 1e-12))

def compute_cosine_similarity(prediction, target):
    """
    Compute cosine similarity
    """
    prediction = np.asarray(prediction)
    target = np.asarray(target)

    num = np.dot(prediction, target)
    den = np.linalg.norm(prediction) * np.linalg.norm(target)
    return num/(den + 1e-12)

def compute_pearson(prediction, target):
    """
    Compute Pearson correlation coefficient.
    """
    prediction = np.asarray(prediction)
    target = np.asarray(target)
    return np.corrcoef(prediction, target)[0,1]

def compute_sam(prediction, target):
    """
    Compute Spectral Angle Mapper.
    """
    prediction = np.asarray(prediction)
    target = np.asarray(target)

    cos_theta = (np.dot(prediction,target)/ (np.linalg.norm(prediction) * np.linalg.norm(target) + 1e-12))
    cos_theta = np.clip(cos_theta, -1, 1)

    return np.arccos(cos_theta)

def compute_snr(prediction, target):
    """
    Compute Signal-to-Noise Ratio.
    """
    signal_power = np.mean(target**2)

    noise_power = np.mean((target - prediction)**2)

    return 10*np.log10(signal_power / (noise_power + 1e-12))

def compute_psnr(prediction, target):
    """
    Compute Peak Signal-to-Noise Ratio.
    """
    mse = compute_mse( prediction, target)
    peak = np.max(target)

    return 20*np.log10(peak / (np.sqrt(mse) + 1e-12))

# raman_metrics
# ==========================================================
from scipy.signal import find_peaks as scipy_find_peaks
from scipy.signal import peak_widths

def find_peaks(intensity, prominence=0.05):
    """
    Detect Raman peaks, importance to keep track of prominence.
    """
    peaks, properties = scipy_find_peaks(intensity, prominence)
    return peaks, properties

def peak_position_error(prediction, target):
    """
    Measure Raman peak position error by furtherest capture peak.
    """
    pred_peaks, _ = find_peaks(prediction)
    target_peaks, _ = find_peaks(target)

    if len(pred_peaks) == 0:
        return np.inf
    if len(target_peaks) == 0:
        return np.inf
    n = min(len(pred_peaks), len(target_peaks))

    return np.mean(np.abs(pred_peaks[:n] - target_peaks[:n]))

def peak_height_error(prediction, target):
    """
    Measure Raman peak height error.
    """
    pred_peaks, _ = find_peaks(prediction)
    target_peaks, _ = find_peaks(target)
    n = min(len(pred_peaks), len(target_peaks))

    if n == 0: return np.inf

    pred_height = prediction[pred_peaks[:n]]
    target_height = target[target_peaks[:n]]

    return np.mean(np.abs(pred_height - target_height))

def peak_width_error(prediction, target):
    """
    Measure Raman peak width error.
    """
    pred_peaks, _ = find_peaks(prediction)
    target_peaks, _ = find_peaks(target)

    if len(pred_peaks) == 0:
        return np.inf
    if len(target_peaks) == 0:
        return np.inf

    pred_widths = peak_widths(prediction, pred_peaks)[0]
    target_widths = peak_widths(target, target_peaks)[0]
    n = min(len(pred_widths), len(target_widths))

    return np.mean(
        np.abs(pred_widths[:n] - target_widths[:n]))

def peak_area_error(prediction, target, peak_mask):
    """
    Measure Raman peak area error.
    """
    pred_area = np.trapezoid(prediction[peak_mask])
    target_area = np.trapezoid(target[peak_mask])

    return np.abs(pred_area - target_area)

def signature_energy(signature):
    """
    Compute extracted Raman signature energy.
    """
    return np.sum(signature**2)

def background_error(prediction, target):
    """
    Measure reconstructed background error.
    """
    return compute_rmse(prediction, target)

def residual_error(residual):
    """
    Measure residual reconstruction error.
    """
    return np.mean(np.abs(residual))
