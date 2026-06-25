# augmentation.py
# ==========================================================
import numpy as np
from scipy.ndimage import gaussian_filter1d

def add_gaussian_noise(intensity, mu=0, sigma=0.2):
    """
    Add Gaussian noise.
    """
    noise = np.random.normal(mu, sigma*np.max(intensity), size=intensity.shape)

    return intensity + noise

def add_poisson_noise(intensity):
    """
    Add Poisson noise.
    """
    # Calculated scaled version
    scaled = intensity - intensity.min()
    scaled /= (scaled.max() + 1e-12)
    scaled *= 1000
    # Noise
    noisy = np.random.poisson(scaled)
    return noisy.astype(np.float32)

def add_shot_noise(intensity):
    """
    Simulate photon shot noise.
    """
    noise = np.random.normal(0,np.sqrt(np.abs(intensity)))
    return intensity + noise

def add_cosmic_ray_noise(intensity, num_spike=5, mag=10):
    """
    Simulate cosmic ray spikes.
    """
    output = intensity.copy()
    idx = np.random.randint(0, len(output), num_spike)  # Gen Spike Locations
    output[idx] += (mag*np.max(output))
    return output

def add_fluorescence_background(shift, intensity, strength=0.5):
    """
    Simulate fluorescence background.
    """
    baseline = (strength*(shift-shift.min())**2)
    baseline /= baseline.max()
    baseline *= intensity.max()

    return intensity + baseline

def random_shift(shift, max_shift=5):
    """
    Randomly shift Raman peaks.
    """
    offset = np.random.uniform(-max_shift, max_shift)
    return shift + offset

def random_scaling(intensity, low=0.8, high = 1.2):
    """
    Randomly scale intensity.
    """
    scale = np.random.uniform(low, high)

    return intensity * scale

def random_baseline_drift(shift, intensity):
    """
    Simulate baseline drift.
    """
    a = np.random.uniform(-1e-5, 1e-5)
    b = np.random.uniform(-0.01, 0.01)

    drift = (a*shift**2 + b*shift)
    return intensity + drift

def random_peak_broadening(intensity):
    """
    Simulate broadened Raman peaks.
    """
        # Calculate sigma
    sigma = np.random.uniform(0.5, 2.5)
    return gaussian_filter1d(intensity, sigma)

def estimate_snr():
    pass

def aug_pipeline():
    pass