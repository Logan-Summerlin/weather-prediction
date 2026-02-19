"""Shared utility helpers."""

import numpy as np


def to_numpy(arr) -> np.ndarray:
    """Convert input to a 1-D float64 numpy array."""
    return np.asarray(arr, dtype=np.float64).ravel()
