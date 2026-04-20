import numpy as np
import pandas as pd

def safe_prepare_data(x, y=None, paired=False, drop_nans=True):
    """Validates inputs, converts to numpy arrays, and handles missing values."""
    
    x = np.asarray(x, dtype=float)
    if y is not None:
        y = np.asarray(y, dtype=float)

    # Handle NaNs
    if drop_nans:
        if y is not None and paired:
            mask = ~np.isnan(x) & ~np.isnan(y)
            x = x[mask]
            y = y[mask]
        else:
            x = x[~np.isnan(x)]
            if y is not None:
                y = y[~np.isnan(y)]
    else:
        if np.isnan(x).any() or (y is not None and np.isnan(y).any()):
            raise ValueError("Input contains NaNs and drop_nans is set to False.")

    if len(x) == 0 or (y is not None and len(y) == 0):
        raise ValueError("Array is empty after dropping NaNs.")
        
    if paired and y is not None:
        if len(x) != len(y):
            raise ValueError("Paired data must have exactly the same number of observations.")

    return x, y
