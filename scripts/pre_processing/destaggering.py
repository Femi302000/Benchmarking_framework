import numpy as np



def destagger(field: np.ndarray,pixel_shift_by_row: np.ndarray) -> np.ndarray:
    """Apply per-row shifts to destagger a 2D lidar field."""
    dest = np.zeros_like(field)
    for u, shift in enumerate(pixel_shift_by_row):
        dest[u, :] = np.roll(field[u, :], shift)
    return dest




