"""
Dataset generation file for strictly decreasing marginal benefits.
There is no neeed to execute this file directly it is imported by createdatasets.py.
-----------------------------------------------
Written by Olin Gilster at the TangAI Lab, 2026
-----------------------------------------------
"""

import numpy as np

"""
Goes into \datasets\strictly_increasing and flips the strictly increasing datasets to make strictly decreasing datasets.
returns the marginals
"""

import os

def generatestrictlydecreasing(
    r: int,
    base_dir: str = "datasets",
):
    """
    Loads the r-th strictly increasing dataset and mirrors it
    to produce strictly decreasing marginal benefits.

    Returns:
        delta_dec : (n, L) ndarray
    """
    strict_dir = os.path.join(base_dir, "strictly_increasing")

    path = os.path.join(strict_dir, f"delta_{r:03d}.csv")
    delta_inc = np.loadtxt(path, delimiter=",")

    # Mirror levels: (n, L) -> (n, L)
    delta_dec = delta_inc[:, ::-1]

    return delta_dec

