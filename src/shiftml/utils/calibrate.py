import numpy as np

def calibrate_ensemble(X, frame):
    """Calibrates ensemble results about the mean,
    by a calibration factor
    """
    Xmean = np.mean(X, axis=-1, keepdims=True)
    diff = X - Xmean

    Z = frame.get_atomic_numbers()

    alphas = 1.5
    
    diff *= alphas
    Xcalibrated = X + diff

    return Xcalibrated