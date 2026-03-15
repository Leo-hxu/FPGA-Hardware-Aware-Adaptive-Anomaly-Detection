"""
feature_utils.py
Utility functions for sliding-window segmentation and feature extraction.
"""

import numpy as np


def sliding_windows(signal: np.ndarray, window_size: int, stride: int):
    """
    Split a 1D time-series signal into overlapping windows.
    Returns shape: [num_windows, window_size]
    """
    windows = []
    n = len(signal)

    if n < window_size:
        return np.empty((0, window_size), dtype=np.float32)

    for start in range(0, n - window_size + 1, stride):
        end = start + window_size
        windows.append(signal[start:end])

    return np.asarray(windows, dtype=np.float32)


def extract_features_from_window(window: np.ndarray) -> np.ndarray:
    """
    Input:
        window: shape [window_size]
    Output:
        features: shape [8]

    The 8 extracted features are:
        mean, max, min, peak_to_peak,
        mean_abs_diff, energy, variance, slope
    """
    mean_val = np.mean(window)
    max_val = np.max(window)
    min_val = np.min(window)
    peak_to_peak = max_val - min_val

    diffs = np.diff(window)
    mean_abs_diff = np.mean(np.abs(diffs)) if len(diffs) > 0 else 0.0

    energy = np.mean(window ** 2)
    variance = np.var(window)

    # Slope from a first-order linear fit.
    x = np.arange(len(window), dtype=np.float32)
    if len(window) > 1:
        slope = np.polyfit(x, window, deg=1)[0]
    else:
        slope = 0.0

    features = np.array([
        mean_val,
        max_val,
        min_val,
        peak_to_peak,
        mean_abs_diff,
        energy,
        variance,
        slope
    ], dtype=np.float32)

    return features
