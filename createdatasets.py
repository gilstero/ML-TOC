"""
This script generates datasets for strictly increasing and nonnegative marginals scenarios.
Each dataset consists of R replications, each with n samples and L levels.
The generated datasets are saved as CSV files in their respective directories.
I reccomend that you do not run this script twice AND keep the number of levels (L) < 4 and the number of samples (n) < 1000
to avoid excessive computation time and storage usage.
-----------------------------------------------
Written by Olin Gilster at the TangAI Lab, 2026
-----------------------------------------------
"""

# main file for creating datasets
from strictlyincreasing import *
from nonnegativemarginals import *
from strictlydecreasing import *
import os

# DO NOT RUN AGAIN IF DATASETS ALREADY GENERATED

# joining paths for saving datasets
BASE_DIR = "datasets"
STRICT_DIR = os.path.join(BASE_DIR, "strictly_increasing")
NONNEG_DIR = os.path.join(BASE_DIR, "nonnegative")
DECREASING_DIR = os.path.join(BASE_DIR, "strictly_decreasing")
os.makedirs(DECREASING_DIR, exist_ok=True)
os.makedirs(STRICT_DIR, exist_ok=True)
os.makedirs(NONNEG_DIR, exist_ok=True)

R = 100 # number of data replications
L = 4 # number of levels
n = 1000 # number of samples

# ----------------------------
# Strictly Increasing Dataset
# ----------------------------
for r in range(R):
    _, delta, _ = generatestrictlyincreasing(
        n=n,
        L=L,
        sigma=0.5,
        seed=r,
        gap=1.0,
    )

    path = os.path.join(STRICT_DIR, f"delta_{r:03d}.csv")
    np.savetxt(path, delta, delimiter=",")

# ----------------------------
# Nonnegative Marginals Dataset
# ----------------------------
for r in range(R):
    _, delta, _ = generatenonnegative(
        n=n,
        L=L,
        sigma=0.5,
        seed=r,
    )

    path = os.path.join(NONNEG_DIR, f"delta_{r:03d}.csv")
    np.savetxt(path, delta, delimiter=",")


# ----------------------------
# Strictly Decreasing Dataset
# ----------------------------
for r in range(R):
    delta = generatestrictlydecreasing(r)

    path = os.path.join(DECREASING_DIR, f"delta_{r:03d}.csv")
    np.savetxt(path, delta, delimiter=",")

    
print("Datasets generated and saved.")

    