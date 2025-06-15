import numpy as np

def check_shapes(X, y):
    assert isinstance(X, np.ndarray), "X must be a numpy array"
    assert isinstance(y, np.ndarray), "y must be a numpy array"
    assert X.ndim == 3, f"X should be (N, C, T), got {X.shape}"
    assert y.ndim == 1, f"y should be (N,), got {y.shape}"
    assert X.shape[0] == y.shape[0], "X and y must have the same number of samples"
