# research question 1

"""
rq1.py generates and saves rankings for various policies on datasets of marginal benefits.
For rq1.py to run properly the datasets must be given to the main loop in the correct directory structure.
-----------------------------------------------
Written by Olin Gilster at the TangAI Lab, 2026
-----------------------------------------------
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Tuple
import pulp

import numpy as np
from tqdm import tqdm

"""
---------
Utilities
---------
"""

DELTA_RE = re.compile(r"delta_(\d{3})\.csv$")

def load_delta(path: str | Path) -> np.ndarray:
    """Load an (n, L) marginal benefit matrix from CSV."""
    delta = np.loadtxt(path, delimiter=",")
    if delta.ndim != 2:
        raise ValueError(f"delta must be 2D, got shape={delta.shape} from {path}")
    return delta

def save_ranking(out_path: str | Path, ranking: np.ndarray) -> None:
    """
    Save ranking as CSV:
      ranking is (B, 2) where each row is (i, l) chosen at step t.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_path, ranking, delimiter=",", fmt="%d")

def list_delta_files(dataset_dir: str | Path) -> list[Path]:
    dataset_dir = Path(dataset_dir)
    files = sorted([p for p in dataset_dir.iterdir() if p.is_file() and DELTA_RE.search(p.name)])
    if not files:
        raise FileNotFoundError(f"No delta_###.csv files found in {dataset_dir}")
    return files

def file_id_from_name(name: str) -> str:
    m = DELTA_RE.search(name)
    if not m:
        raise ValueError(f"Unexpected delta filename: {name}")
    return m.group(1)

def save_allocations_by_budget(out_path: str | Path, allocs_by_budget: np.ndarray) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_path, allocs_by_budget, delimiter=",", fmt="%d")

def pag_NF(delta: np.ndarray) -> np.ndarray:
    """
    PAG with previous ranking subsetting policy
    """
    n, L = delta.shape
    B = n * L

    ranking = np.zeros((B, 2), dtype=int)

    # next feasible level for each individual
    next_level = np.zeros(n, dtype=int)

    for b in range(B):
        best_i = -1
        best_val = -np.inf

        # scan feasible frontier
        for i in range(n):
            l = next_level[i]
            if l < L and delta[i, l] > best_val:
                best_val = delta[i, l]
                best_i = i

        l = next_level[best_i]
        ranking[b] = (best_i, l)
        next_level[best_i] += 1  # unlock next level

    return ranking

def pag_F(delta: np.ndarray) -> np.ndarray:
    n, L = delta.shape
    B = n * L

    prefix = np.zeros((n, L + 1), dtype=np.float64)
    prefix[:, 1:] = np.cumsum(delta, axis=1)

    dp_prev = np.full(B + 1, -np.inf, dtype=np.float64)
    dp_prev[0] = 0.0

    choice = np.zeros((n, B + 1), dtype=np.uint8)

    for i in range(n):
        best = dp_prev.copy()
        best_m = np.zeros(B + 1, dtype=np.uint8)

        for m in range(1, L + 1):
            cand = np.full(B + 1, -np.inf, dtype=np.float64)
            cand[m:] = dp_prev[:-m] + prefix[i, m]

            better = cand > best
            best = np.where(better, cand, best)
            best_m = np.where(better, m, best_m)

        choice[i, :] = best_m
        dp_prev = best

    allocs_by_budget = np.zeros((B, n * L), dtype=np.uint8)

    for b in range(1, B + 1):
        bb = b
        alloc_flat = np.zeros(n * L, dtype=np.uint8)

        for i in range(n - 1, -1, -1):
            m = int(choice[i, bb])
            bb -= m
            if m > 0:
                start = i * L
                alloc_flat[start:start + m] = 1

        allocs_by_budget[b - 1, :] = alloc_flat

    return allocs_by_budget

def prp(delta: np.ndarray) -> np.ndarray:
    """
    PRP (Precedence-Respecting Prefix) policy:
      Treat every level fully before moving to the next level
    """
    n, L = delta.shape
    B = n * L

    ranking = np.zeros((B, 2), dtype=int)
    k = 0

    for l in range(L):
        ind_order = np.argsort(-delta[:, l])  # descending for this level
        for i in ind_order:
            ranking[k, 0] = i
            ranking[k, 1] = l
            k += 1

    return ranking

def gs(delta: np.ndarray) -> np.ndarray:
    """
    Global-score policy:
    1) Compute total score for each unit i: sum over l of delta[i, l]
    2) Rank units i by total score descending
    3) For each unit in ranked order, add all (i, l) pairs
    """
    n, L = delta.shape
    B = n * L

    scores = delta.sum(axis=1)              # (n,)
    ind_order = np.argsort(-scores)         # descending

    ranking = np.zeros((B, 2), dtype=int)
    k = 0
    for i in ind_order:
        for l in range(L):
            ranking[k, 0] = i
            ranking[k, 1] = l
            k += 1
    return ranking

def greedy(delta: np.ndarray) -> np.ndarray:
    """
    Greedy / Marginal-Effect policy
    """
    n, L = delta.shape
    B = n * L

    flat = delta.reshape(-1)
    order = np.argsort(-flat)  # descending by marginal benefit

    ranking = np.zeros((B, 2), dtype=int)
    ranking[:, 0] = order // L
    ranking[:, 1] = order % L
    return ranking

"""
----------
Executable
----------
"""

def run_rq1_rankings(
    dataset_dir: str | Path,
    results_dir: str | Path = "resultsrq1",
    n_mc_random: int = 200,
    random_seed_base: int = 0,
) -> None:
    """
    For each delta_###.csv in dataset_dir:
      - save greedy ranking to resultsrq1/rankings/###_greedy.csv
      - compute and save random baseline ATE curve to resultsrq1/random_baseline/###_random.txt
    """
    dataset_dir = Path(dataset_dir)
    results_dir = Path(results_dir)

    out_rankings = results_dir / "rankings"
    out_random = results_dir / "random_baseline"
    out_rankings.mkdir(parents=True, exist_ok=True)
    out_random.mkdir(parents=True, exist_ok=True)

    delta_files = list_delta_files(dataset_dir)

    # tqdm progress bar
    for p in tqdm(delta_files, desc="Processing datasets", unit="file"):
        idx = file_id_from_name(p.name)  # "000"
        delta = load_delta(p)

        # for speed purposes I would reccomend only running one of the methods at a time.
        # comment out the methods you do not wish to run.

        # # 1a) Greedy ranking
        # r_greedy = greedy(delta)
        # save_ranking(out_rankings / f"{idx}_greedy.csv", r_greedy)

        # 1b) Global-score ranking
        # r_gs = gs(delta)
        # save_ranking(out_rankings / f"{idx}_gs.csv", r_gs)

        # 1c) Precedence-Respecting Prefix ranking
        # r_prp = prp(delta)
        # save_ranking(out_rankings / f"{idx}_prp.csv", r_prp)

        # 1d) PAG-F
        r_pag_f = pag_F(delta)
        save_allocations_by_budget(out_rankings / f"{idx}_pag_f.csv", r_pag_f)

        # 1e) PAG-NF
        # r_pag_nf = pag_NF(delta)
        # save_ranking(out_rankings / f"{idx}_pag_nf.csv", r_pag_nf)

        # 2) Random baseline ATE
        # ate_rand = random_baseline_ate(
        #     delta,
        #     n_mc=n_mc_random,
        #     seed=random_seed_base + int(idx),
        # )
        # save_random_baseline(out_random / f"{idx}_random.txt", ate_rand)


"""
---------------------------------
Functions for the random baseline
---------------------------------
"""

def random_baseline_ate(
    delta: np.ndarray,
    n_mc: int = 200,
    seed: int | None = None,
) -> np.ndarray:
    """
    Monte Carlo random baseline under precedence, returning AVG ATE per budget increment.

    Output:
        baseline_avg : (B,) where baseline_avg[b-1] = E[ cumulative_ate(b) / b ].
    """
    n, L = delta.shape
    B = n * L
    rng = np.random.default_rng(seed)

    baseline_sum = np.zeros(B, dtype=float)

    for _ in range(n_mc):
        next_level = np.zeros(n, dtype=np.int16)  # next feasible level index 0..L-1
        cum = 0.0

        for b in range(1, B + 1):
            feasible = np.flatnonzero(next_level < L)
            i = int(rng.choice(feasible))
            l = int(next_level[i])

            cum += float(delta[i, l])
            next_level[i] += 1

            baseline_sum[b - 1] += cum / b  # divide by budget increments

    return baseline_sum / n_mc

def save_random_baseline(path: str | Path, baseline_avg: np.ndarray):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("# budget mean_random_avg_ate\n")
        for b, val in enumerate(baseline_avg, start=1):
            f.write(f"{b} {val:.6f}\n")

"""
----------
Main loop
----------
"""

if __name__ == "__main__":
    # uncomment one of the following to run RQ1 rankings on that dataset type

    # Strictly increasing marginal benefits
    # datasets/strictly_increasing/delta_000.csv, ...
    # run_rq1_rankings(dataset_dir="datasets/strictly_increasing", results_dir="resultsrq1")

    # Non-negative marginal benefits
    # datasets/nonnegative/delta_000.csv, ...
    # run_rq1_rankings(dataset_dir="datasets/nonnegative", results_dir="resultsrq1nn")

    # Strictly decreasing marginal benefits
    run_rq1_rankings(dataset_dir="datasets/strictly_decreasing", results_dir="resultsrq1sd")