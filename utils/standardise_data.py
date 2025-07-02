from sklearn.preprocessing import StandardScaler
import numpy as np

def standardize_eeg_data(X_train, X_val, X_test):
    """
    Standardize EEG data properly using training statistics
    """
    # Reshape for StandardScaler: (samples, features)
    n_train, n_channels, n_timepoints = X_train.shape
    X_train_flat = X_train.reshape(n_train, -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    # Fit scaler on training data only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat)
    
    # Apply same transformation to val/test
    X_val_scaled = scaler.transform(X_val_flat)
    X_test_scaled = scaler.transform(X_test_flat)
    
    # Reshape back
    X_train_scaled = X_train_scaled.reshape(n_train, n_channels, n_timepoints)
    X_val_scaled = X_val_scaled.reshape(X_val.shape[0], n_channels, n_timepoints)
    X_test_scaled = X_test_scaled.reshape(X_test.shape[0], n_channels, n_timepoints)
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler
