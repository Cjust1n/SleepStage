"""
utils.py

Utility functions for feature extraction.

Author: Justin
"""

import numpy as np


def magnitude(x, y, z):
    """
    Compute acceleration magnitude.
    """

    return np.sqrt(x**2 + y**2 + z**2)


def rms(signal):
    """
    Root Mean Square.
    """

    return np.sqrt(np.mean(signal**2))


def signal_energy(signal):
    """
    Signal Energy.
    """

    return np.sum(signal**2)


def zero_crossing(signal):
    """
    Zero Crossing Rate.
    """

    centered = signal - np.mean(signal)

    return np.sum(
        centered[:-1] * centered[1:] < 0
    )


def movement_count(signal, threshold=0.05):
    """
    Number of movement transitions.
    """

    diff = np.abs(np.diff(signal))

    return np.sum(diff > threshold)


def variance(signal):

    return np.var(signal)


def std(signal):

    return np.std(signal)


def mean(signal):

    return np.mean(signal)