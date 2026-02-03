"""
timing.py is an inidvidual file that is designed to measure the time it takes to rank datasets of varying sizes using different policies.
It generates one type of dataset (strictly increasing marginals) for n = 500, 1000, 2000, 4000, 8000 and L = 4.
It then ranks the policies using the generated datasets and records the time that is taken for each.
-----------------------------------------------
Written by Olin Gilster at the TangAI Lab, 2026
-----------------------------------------------
"""

import numpy as np
import os
import time
import sys
import matplotlib.pyplot as plt

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # Thesis/
sys.path.append(str(ROOT))

from tqdm import tqdm
from strictlyincreasing import *
from rq1 import greedy, pag_F, pag_NF, gs, prp

# create data specific for timing
# we will use n=500, 1000, 2000, 4000, 8000 and L = 4
n_values = [500, 1000, 2000, 4000, 8000]
L = 4

BASE_DIR = "datasets"
STRICT_DIR = os.path.join(BASE_DIR, "timing_datasets")

policies = {"pag_NF", "greedy", "gs", "prp", "pag_F"}


"""
Execute and time the rankings
"""
def time_rankings(delta: np.ndarray) -> dict[str, float]:
    times: dict[str, float] = {}

    # name -> function
    fns = {
        "greedy": greedy,
        "gs": gs,
        "prp": prp,
        "pag_F": pag_F,
        "pag_NF": pag_NF,
    }

    
    for name in tqdm(sorted(policies), desc="  Policies", leave=False):
        fn = fns[name]
        t0 = time.perf_counter()
        _ = fn(delta)  # assumes your rq1 policy funcs take delta as input
        t1 = time.perf_counter()
        times[name] = t1 - t0

    return times
    


"""
Main function
"""
def main():
    results: dict[str, list[float]] = {p: [] for p in policies}

    for n in tqdm(n_values, desc="Datasets (n)", unit="dataset"):
        # generate dataset
        _, delta, _ = generatestrictlyincreasing(
        n=n,
        L=L,
        sigma=0.5,
        seed=0,
        gap=1.0,
        )

        path = os.path.join(STRICT_DIR, f"delta_n{n}.csv")
        np.savetxt(path, delta, delimiter=",")
        print(f"[saved] {path} shape={delta.shape}")

        times = time_rankings(delta)

        for p in policies:
            results[p].append(times[p])

        print(f"[timing] n={n} " + " ".join([f"{k}={v:.4f}s" for k, v in times.items()]))

    DISPLAY_PLOT = {
    "prp": "PRPP",
    "gs": "GSP",
    "greedy": "MEP",
    "pag_F": "PAG-F",
    "pag_NF": "PAG-NF",
    }

    PLOT_ORDER = ["prp", "gs", "greedy", "pag_NF", "pag_F"]
    colors = plt.cm.Blues(np.linspace(0.35, 0.85, len(PLOT_ORDER)))

    x = np.arange(len(n_values))
    width = 0.15

    plt.figure(figsize=(10, 5))

    for j, (p, c) in enumerate(zip(PLOT_ORDER, colors)):
        plt.bar(
            x + j * width,
            results[p],
            width,
            label=DISPLAY_PLOT[p],
            color=c,
            edgecolor="black",
            linewidth=0.5,
        )

    # graphing details so everything looks nice
    plt.xticks(x + width * (len(PLOT_ORDER) - 1) / 2, n_values)
    plt.xlabel("Dataset size (n)", fontsize=11)
    plt.yscale("log")
    plt.ylabel("Runtime (seconds, log scale)", fontsize=11)
    plt.title("Policy Runtime vs Dataset Size", fontsize=13)

    plt.legend(title="Policy", frameon=False)
    plt.grid(axis="y", linestyle=":", alpha=0.4)
    plt.show()



if __name__ == "__main__":
    main()