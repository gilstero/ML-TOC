import numpy as np

def generateheterogeneousstrictlyincreasing(n, L, sigma=0.5, seed=None, gap=1.0):
    """
    Generates a strictly increasing marginal benefit dataset with cross-individual heterogeneity.

    Each individual i has a latent scale s_i, so magnitudes differ across individuals,
    but monotonicity is enforced within each individual.

    Returns:
        X (None placeholder),
        delta (n x L matrix of marginal benefits),
        None (placeholder for compatibility)
    """
    if seed is not None:
        np.random.seed(seed)

    # latent scale per individual (positive)
    scales = np.random.lognormal(mean=0.0, sigma=0.5, size=n)

    delta = np.zeros((n, L))

    for i in range(n):
        # initial level
        base = np.random.normal(loc=scales[i], scale=sigma)
        delta[i, 0] = max(base, 0.0)

        # enforce strictly increasing structure
        for l in range(1, L):
            raw = np.random.normal(loc=scales[i], scale=sigma)
            delta[i, l] = max(raw, delta[i, l-1] + gap)

    return None, delta, None