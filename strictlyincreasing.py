"""
Dataset generation file for strictly increasing marginal benefits.
There is no neeed to execute this file directly it is imported by createdatasets.py.
-----------------------------------------------
Written by Olin Gilster at the TangAI Lab, 2026
-----------------------------------------------
"""

import numpy as np

def generatestrictlyincreasing(n:int, L:int, sigma:float, seed: int, gap:float = 1.0, mu:float = 0.0, d:int = 10):
    """
    X : (n, d) ndarray
        Covariates
    delta : (n, L) ndarray
        Marginal benefits
    """

    if seed is not None:
        np.random.seed(seed)

    # Covariates
    X = np.random.normal(0, 1, size=(n, d))

    # Raw draws
    Z = np.random.normal(mu, sigma, size=(n, L))

    delta = np.zeros((n, L))

    # Level 1
    delta[:, 0] = np.maximum(0.0, Z[:, 0])

    # Enforce strict increase
    for l in range(1, L):
        delta[:, l] = np.maximum(Z[:, l], delta[:, l - 1] + gap) # this is forcing a strict increase of at least 'gap'

    costs = np.ones(L)

    return X, delta, costs