import numpy as np

def to_gasf(x:np.array): # Gramian Angular Summation Field
    """
    Parameters:
    x (array-like): The input time series data. Should be a 1D array or list.
    
    Returns:
    numpy.ndarray: The transformed Gramian Angular Summation Field.
    """
    # Normalize the time series data
    x_norm = (x - np.min(x)) / (np.max(x) - np.min(x))  # Scaled to [0, 1]
    x_scaled = 2 * x_norm - 1  # Rescaled to [-1, 1]
    # Polar encoding
    phi = np.arccos(x_scaled)
    return np.cos(np.add.outer(phi, phi))

def to_gadf(x:np.array): # Gramian Angular Difference Field
    """
    Parameters:
    x (array-like): The input time series data. Should be a 1D array or list.
    
    Returns:
    numpy.ndarray: The transformed Gramian Angular Difference Field.
    """
    # Normalize the time series data
    x_norm = (x - np.min(x)) / (np.max(x) - np.min(x))  # Scaled to [0, 1]
    x_scaled = 2 * x_norm - 1  # Rescaled to [-1, 1]
    # Polar encoding
    phi = np.arccos(x_scaled)
    return np.sin(np.subtract.outer(phi, phi))