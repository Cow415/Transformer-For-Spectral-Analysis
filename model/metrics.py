import numpy as np

def energy_loss(orig, processed):
    """
    Calculates the residual energy loss between an original (raw) spectrum and a processed spectrum. 
    It measures the total signal power lost during preprocessing relative to the original signal power.
    """
    orig_arr = np.asarray(orig)
    processed_arr = np.asarray(processed)
    
    residual_energy = np.sum((orig_arr - processed_arr) ** 2)
    orig_energy = np.sum(orig_arr ** 2)
    
    # Avoid division by zero
    if orig_energy == 0:
        return 0.0
        
    return residual_energy / orig_energy
