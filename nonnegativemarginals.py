"""
Dataset generation file for non negative marginal benefits.
There is no neeed to execute this file directly it is imported by createdatasets.py.
-----------------------------------------------

-----------------------------------------------
"""

# code for non-negative marginals only
import numpy as np

def generatenonnegative(n:int, L:int, sigma:float, seed: int, gap:float = 1.0, mu:float = 0.0, d:int = 10):
    """
    X : (n, d) ndarray
        Covariates
    delta : (n, L) ndarray
        Marginal benefits (nonnegative)
    """

    rng = np.random.default_rng(seed)

    # Covariates
    X = rng.normal(0, 1, size=(n, d))

    # Raw draws; allow nonmonotone patterns across levels
    Z = rng.normal(mu, sigma, size=(n, L))

    # Enforce nonnegativity (half-normal style)
    delta = np.maximum(0.0, Z)

    costs = np.ones(L)
    return X, delta, costs