import numpy as np

def augment_eeg_data(X, noise_factor=0.1):
    """Add small amount of noise for augmentation"""
    noise = np.random.normal(0, noise_factor, X.shape)
    return X + noise
