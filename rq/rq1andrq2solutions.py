# rq1 graphing and area output + rq2 terminal output

"""
RQ1 and RQ2: ML-AUTOC graphing and area output + terminal summary
Run the file to generate ML-AUTOC plots and area summaries for specified datasets and policies.
Edit the main block at the bottom to customize dataset selection and output options.
-----------------------------------------------

-----------------------------------------------
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# -------------------
# Config / labels
# -------------------

DISPLAY = {
    "pag_f": "PAG*",
    "pag_nf": "PAG",
    "greedy": "PGP",   # was MEP
    "gs": "IFP",       # was GSP
    "prp": "LFP",      # was PRPP
}
# Policies that MUST be executed with precedence feasibility
PRECEDENCE_EXEC_POLICIES = {"pag_nf", "gs", "prp"}

# Policies for which ML-AUTOC (area) is valid to report
# area is calculated using the ATE - Random baseline for that specific dataset.
AREA_VALID_POLICIES = {"pag_nf", "greedy", "gs", "prp"}

# -------------------
# Plot styles (consistent across dataset types)
# -------------------

# Fixed policy order -> stable color assignment across runs/datasets
POLICY_ORDER = ["pag_nf", "pag_f", "greedy", "gs", "prp"]
_DEFAULT_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]

POLICY_COLORS = {
    p: _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)]
    for i, p in enumerate(POLICY_ORDER)
}

# Line styles:
# - MEP is infeasible so dashed
POLICY_LINESTYLE = {
    "greedy": "--", 
}

def _style_for_policy(policy: str) -> tuple[str | None, str]:
    color = POLICY_COLORS.get(policy, None)
    ls = POLICY_LINESTYLE.get(policy, "-")
    return color, ls

# -------------------
# IO helpers
# -------------------

DELTA_RE = re.compile(r"delta_(\d{3})\.csv$")

def list_dataset_ids(dataset_dir: str | Path) -> List[str]:
    dataset_dir = Path(dataset_dir)
    ids: List[str] = []
    for p in sorted(dataset_dir.iterdir()):
        m = DELTA_RE.search(p.name)
        if m:
            ids.append(m.group(1))
    if not ids:
        raise FileNotFoundError(f"No delta_###.csv files found in {dataset_dir}")
    return ids

def load_delta(dataset_dir: str | Path, idx: str) -> np.ndarray:
    dataset_dir = Path(dataset_dir)
    return np.loadtxt(dataset_dir / f"delta_{idx}.csv", delimiter=",")

def load_ranking_csv(rankings_dir: str | Path, idx: str, policy: str) -> np.ndarray:
    rankings_dir = Path(rankings_dir)
    return np.loadtxt(rankings_dir / f"{idx}_{policy}.csv", delimiter=",", dtype=int)

def _load_random_baseline(path: str | Path, Bmax: int) -> np.ndarray:
    """
    Reads:
      # budget mean_random_ate
      1 1.743249
      2 3.425630
      ...
    Returns baseline array of shape [Bmax] aligned to budgets 1..Bmax.
    """
    path = Path(path)
    baseline = np.zeros(Bmax, dtype=float)

    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            b = int(parts[0])
            v = float(parts[1])
            if 1 <= b <= Bmax:
                baseline[b - 1] = v

    # If file is shorter than Bmax, hold last seen value flat
    last = 0.0
    for k in range(Bmax):
        if baseline[k] != 0.0:
            last = baseline[k]
        else:
            baseline[k] = last

    return baseline

# Given a cumulative allocation-by-budget matrix, compute total benefit, average ATE per budget,
# and the ML-AUTOC curve (average ATE minus the random baseline).
def toc_from_alloc_matrix(alloc: np.ndarray, delta: np.ndarray, baseline: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, L = delta.shape
    Bmax = n * L
    delta_flat = delta.reshape(Bmax)

    if alloc.ndim == 3 and alloc.shape[1:] == (n, L):
        alloc2 = alloc.reshape(Bmax, Bmax)
    elif alloc.ndim == 2 and alloc.shape == (Bmax, Bmax):
        alloc2 = alloc
    else:
        raise ValueError(f"alloc has unexpected shape {alloc.shape}, expected (Bmax,Bmax) or (Bmax,n,L)")

    total = alloc2 @ delta_flat  # total benefit at each budget
    budgets = np.arange(1, Bmax + 1, dtype=float)
    avg = total / budgets
    toc = avg - baseline
    return toc, avg, total

def _load_actions(rankings_dir: str | Path, idx: str, policy: str, L: int) -> np.ndarray:
    r = np.asarray(load_ranking_csv(rankings_dir, idx, policy))

    # ---------- A) 1D encoded actions ----------
    if r.ndim == 1:
        a = r.astype(int)
        i = a // L
        ell_idx = a % L
        return np.stack([i, ell_idx], axis=1)

    # ---------- C) one-hot matrix (Bmax x Bmax) ----------
    # Heuristic: square matrix, much wider than 2 columns
    if r.ndim == 2 and r.shape[1] > 2 and r.shape[0] == r.shape[1]:
        # choose column index per row
        a = np.argmax(r, axis=1).astype(int)  # action index in 0..Bmax-1
        i = a // L
        ell_idx = a % L
        return np.stack([i, ell_idx], axis=1)

    # ---------- B) (i, ell) list ----------
    if r.ndim == 2 and r.shape[1] >= 2:
        i = r[:, 0].astype(int)
        ell = r[:, 1].astype(int)

        # Debug-friendly: infer encoding from observed range
        ell_min, ell_max = int(ell.min()), int(ell.max())

        # Case: ell in 0..L  (0 is dummy/no-op)
        if ell_min == 0 and ell_max == L:
            mask = ell > 0                # drop dummy rows
            i = i[mask]
            ell = ell[mask]
            ell_idx = ell - 1             # 1..L -> 0..L-1
            return np.stack([i, ell_idx], axis=1)

        # Case: ell in 1..L (all real)
        if ell_min >= 1 and ell_max <= L:
            ell_idx = ell - 1
            return np.stack([i, ell_idx], axis=1)

        # Case: ell in 0..L-1 (already zero indexed)
        if ell_min >= 0 and ell_max <= L - 1:
            ell_idx = ell
            return np.stack([i, ell_idx], axis=1)

        raise ValueError(
            f"Unrecognized ell encoding for {idx}_{policy}: "
            f"ell_min={ell_min}, ell_max={ell_max}, expected 0..{L} or 1..{L} or 0..{L-1}"
        )

    raise ValueError(f"Unrecognized ranking format for {idx}_{policy}: shape={r.shape}")

# -------------------
# Test Graphing and Printout
# plot_MLAUTOC_randomchoice is a function that selects a random dataset from the specificed subdirectory to graph.
# This functions uses helper funtions such as load_delta, load_ranking_csv, load_allocs, _load_random_baseline, toc_from_alloc_matrix, and _load_actions.
# There is no use case for running this function unless you want to debug a specific dataset.
# -------------------

def plot_MLAUTOC_randomchoice(
    dataset_subdir: str = "nonnegative",
    results_dir: str = "resultsrq1nn",
    dataset_idx: str | None = None,
    debug: bool = True,
    debug_first_k: int = 30,
    debug_range: tuple[int, int] | None = None,   # (lo, hi) budget range to debug print
    debug_policies: set[str] | None = None, ):
    
    LINESTYLE = { # given for overlapping lines
    "pag_nf": "--",
    "gs": ":",
    }

    # --- all paths to the data ---
    HERE = Path(__file__).resolve().parent
    ROOT = HERE.parent

    dataset_dir = ROOT / "datasets" / dataset_subdir
    rankings_dir = ROOT / results_dir / "rankings"
    baseline_dir = ROOT / results_dir / "random_baseline"

    # Pick dataset if not explicitly provided
    dataset_ids = list_dataset_ids(dataset_dir)
    if dataset_idx is None:
        dataset_idx = np.random.default_rng(0).choice(dataset_ids)

    # Load marginal benefits`
    delta = load_delta(dataset_dir, dataset_idx)
    n, L = delta.shape
    Bmax = n * L

    baseline_path = baseline_dir / f"{dataset_idx}_random.txt"
    baseline = _load_random_baseline(baseline_path, Bmax)

    if debug:
        print(f"\n[Debug] dataset={dataset_idx} n={n} L={L} Bmax={Bmax}")
        print(f"[Debug] baseline first 10: {np.array2string(baseline[:10], precision=4)}")

    policies = ["greedy", "gs", "prp", "pag_f", "pag_nf"]
    areas: dict[str, float] = {} 
    plt.figure(figsize=(8, 5))

    # --- process each policy ---
    for policy in policies:
        if debug_policies is not None and policy not in debug_policies:
            continue
        npy_path = rankings_dir / f"{dataset_idx}_{policy}.npy"
        csv_path = rankings_dir / f"{dataset_idx}_{policy}.csv"

        # ---------- Allocation-matrix policies ----------
        if npy_path.exists():
            alloc = np.load(npy_path)
            # Compute AVG ATE and TOC directly from cumulative allocations
            toc, avg, total = toc_from_alloc_matrix(alloc, delta, baseline)

            if debug:
                print(f"\n--- {policy} ({DISPLAY.get(policy, policy)}) [ALLOC .npy] ---")
                print(f"[Debug] alloc shape={alloc.shape}, dtype={alloc.dtype}")
                rs = (alloc.reshape(Bmax, Bmax) if alloc.ndim == 3 else alloc).sum(axis=1)
                print(f"[Debug] row_sums first 10: {rs[:10].tolist()} (should be 1..10)")

                # Decide which budgets to print
                if debug_range is None:
                    lo, hi = 1, min(debug_first_k, Bmax)
                else:
                    lo, hi = debug_range
                    hi = min(hi, Bmax)
                for b in range(lo - 1, hi):
                    print(f"b={b+1:4d} AVG={avg[b]: .6f} base={baseline[b]: .6f} TOC={toc[b]: .6f}")
            
            areas[policy] = float(np.sum(toc))
            plt.plot(
                toc,
                label=DISPLAY.get(policy, policy),
                linewidth=2,
                linestyle=LINESTYLE.get(policy, "-"),
            )
            continue

        # ---------- Action-list policies ----------
        if not csv_path.exists():
            if debug:
                print(f"[skip] missing {csv_path.name} and no .npy")
            continue

        # fallback: action-list rankings
        actions = _load_actions(rankings_dir, dataset_idx, policy, L=L)
        total = 0.0
        avg = np.zeros(Bmax, dtype=float)

        # Accumulate marginal benefits and convert to AVG ATE
        K = min(Bmax, actions.shape[0])
        for b in range(K):
            i, ell_idx = int(actions[b, 0]), int(actions[b, 1])
            if 0 <= i < n and 0 <= ell_idx < L:
                total += float(delta[i, ell_idx])
            avg[b] = total / (b + 1)

        # Hold final value flat if ranking shorter than Bmax
        if K > 0 and K < Bmax:
            avg[K:] = avg[K - 1]

        toc = avg - baseline
        if policy in AREA_VALID_POLICIES:
            areas[policy] = float(np.sum(toc))

        if debug:
            print(f"\n--- {policy} ({DISPLAY.get(policy, policy)}) [ACTIONS .csv] ---")
            print(f"[Debug] first 5 actions: {actions[:5].tolist()}")
            if debug_range is None:
                lo, hi = 1, min(debug_first_k, Bmax)
            else:
                lo, hi = debug_range
                hi = min(hi, Bmax)
            for b in range(lo - 1, hi):
                print(f"b={b+1:4d} AVG={avg[b]: .6f} base={baseline[b]: .6f} TOC={toc[b]: .6f}")

        plt.plot(
            toc,
            label=DISPLAY.get(policy, policy),
            linewidth=2,
            linestyle=LINESTYLE.get(policy, "-"),
        )

    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Budget")
    plt.ylabel("Avg ATE − Random Baseline (avg)")
    plt.title(f"ML-AUTOC Curves (Dataset {dataset_idx})")
    plt.legend()
    plt.tight_layout()
    plt.show()
    print("\n=== Area (sum of TOC over budgets) ===")
    for policy in policies:
        print(f"{DISPLAY.get(policy, policy)}: {areas[policy]:.6f}")

# -------------------
# plot_MLAUTOC_all_datasets function plots the average ML-AUTOC curves across all datasets.
# This function can be used to visualize the overall performance of different policies under different types of data.
# -------------------

def MLAUTOC_print_summary(
    area_stats: dict[str, dict[str, float]],
    *,
    policies: list[str],
    caption: str = "ML-AUTOC summary across datasets",
    label: str = "tab:mlautoc_summary",
    show_sem: bool = False,
) -> None:
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")

    colspec = "lccc" if show_sem else "lcc"
    lines.append(rf"\begin{{tabular}}{{{colspec}}}")
    lines.append(r"\toprule")

    if show_sem:
        lines.append(r"Policy & Mean ML-AUTOC & Std & SEM \\")
    else:
        lines.append(r"Policy & Mean ML-AUTOC & Std \\")

    lines.append(r"\midrule")

    for policy in policies:
        if policy not in area_stats:
            continue

        name = DISPLAY.get(policy, policy)
        mean = area_stats[policy]["mean"]
        std = area_stats[policy]["std"]

        if show_sem:
            sem = area_stats[policy]["sem"]
            lines.append(rf"{name} & {mean:.6f} & {std:.6f} & {sem:.6f} \\")
        else:
            lines.append(rf"{name} & {mean:.6f} & {std:.6f} \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{table}")

    print("\n" + "\n".join(lines) + "\n")

def _compute_toc_for_policy(
    dataset_idx: str,
    policy: str,
    *,
    dataset_dir: Path,
    rankings_dir: Path,
    baseline_dir: Path,
) -> tuple[np.ndarray, int, int]:
    """
    Returns (toc, n, L) for this dataset+policy.
    toc is length Bmax = n*L.
    """
    delta = load_delta(dataset_dir, dataset_idx)
    n, L = delta.shape
    Bmax = n * L

    baseline_path = baseline_dir / f"{dataset_idx}_random.txt"
    baseline = _load_random_baseline(baseline_path, Bmax)

    npy_path = rankings_dir / f"{dataset_idx}_{policy}.npy"
    csv_path = rankings_dir / f"{dataset_idx}_{policy}.csv"

    # Allocation-matrix policies (.npy)
    if npy_path.exists():
        alloc = np.load(npy_path)
        toc, avg, total = toc_from_alloc_matrix(alloc, delta, baseline)
        return toc, n, L

    # Action-list policies (.csv)
    if csv_path.exists():
        actions = _load_actions(rankings_dir, dataset_idx, policy, L=L)
        total = 0.0
        avg = np.zeros(Bmax, dtype=float)

        K = min(Bmax, actions.shape[0])
        for b in range(K):
            i, ell_idx = int(actions[b, 0]), int(actions[b, 1])
            if 0 <= i < n and 0 <= ell_idx < L:
                total += float(delta[i, ell_idx])
            avg[b] = total / (b + 1)

        if K > 0 and K < Bmax:
            avg[K:] = avg[K - 1]

        toc = avg - baseline
        return toc, n, L

    raise FileNotFoundError(f"Missing ranking for {dataset_idx}_{policy}: no .npy or .csv")

def plot_MLAUTOC_all_datasets(
    dataset_subdir: str = "nonnegative",
    results_dir: str = "resultsrq1nn",
    *,
    policies: list[str] | None = None,
    debug: bool = True,
    title_suffix: str | None = None,
) -> None:
    """
    Also prints a LaTeX table of ML-AUTOC areas (sum of TOC over budgets) mean ± std.
    """
    
    if policies is None:
        policies = ["greedy", "gs", "prp", "pag_f", "pag_nf"]

    # --- paths ---
    HERE = Path(__file__).resolve().parent
    ROOT = HERE.parent
    dataset_dir = ROOT / "datasets" / dataset_subdir
    rankings_dir = ROOT / results_dir / "rankings"
    baseline_dir = ROOT / results_dir / "random_baseline"

    dataset_ids = list_dataset_ids(dataset_dir)
    if debug:
        print(f"\n[Aggregate] subdir={dataset_subdir} datasets={len(dataset_ids)} results_dir={results_dir}")
        print(f"[Aggregate] policies={policies}")

    # We need a common Bmax to average pointwise.
    # If your synthetic setup always uses the same n,L within a subdir, this is perfect.
    # If not, we fall back to truncating to the minimum Bmax observed (safe + simple).
    Bmax_list = []
    for idx in dataset_ids:
        delta = load_delta(dataset_dir, idx)
        n, L = delta.shape
        Bmax_list.append(n * L)
    B_common = int(min(Bmax_list))

    if debug and len(set(Bmax_list)) > 1:
        print(f"[Aggregate] WARNING: varying Bmax across datasets: min={min(Bmax_list)} max={max(Bmax_list)}")
        print(f"[Aggregate] Using B_common=min(Bmax)={B_common} (curves truncated to this length).")

    # Online accumulators for TOC curves:
    # sum_toc[policy] shape (B_common,)
    # sumsq_toc[policy] shape (B_common,)
    sum_toc: dict[str, np.ndarray] = {p: np.zeros(B_common, dtype=float) for p in policies}
    sumsq_toc: dict[str, np.ndarray] = {p: np.zeros(B_common, dtype=float) for p in policies}
    count_toc: dict[str, int] = {p: 0 for p in policies}

    # Area (ML-AUTOC) accumulators per policy
    area_vals: dict[str, list[float]] = {p: [] for p in policies}

    pagf_gt_gs_pct: list[float] = []
    pagf_gt_pagnf_pct: list[float] = []

    # --- main loop over datasets ---
    for di, idx in enumerate(dataset_ids, start=1):
        if debug and (di == 1 or di % 25 == 0 or di == len(dataset_ids)):
            print(f"[Aggregate] processing {di}/{len(dataset_ids)} (dataset={idx})")

        toc_cache: dict[str, np.ndarray] = {}

        for policy in policies:
            try:
                toc, n, L = _compute_toc_for_policy(
                    idx, policy, dataset_dir=dataset_dir, rankings_dir=rankings_dir, baseline_dir=baseline_dir
                )
            except FileNotFoundError as e:
                if debug:
                    print(f"[skip] {e}")
                continue

            toc = np.asarray(toc, dtype=float)
            if toc.shape[0] < B_common:
                padded = np.empty(B_common, dtype=float)
                padded[: toc.shape[0]] = toc
                padded[toc.shape[0] :] = toc[-1]
                toc_use = padded
            else:
                toc_use = toc[:B_common]

            toc_cache[policy] = toc_use
            sum_toc[policy] += toc_use
            sumsq_toc[policy] += toc_use * toc_use
            count_toc[policy] += 1

            # Only report area where valid (your rule)
            if policy in AREA_VALID_POLICIES:
                area_vals[policy].append(float(np.sum(toc_use) / B_common))
        
        if "pag_f" in toc_cache and "gs" in toc_cache:
            pagf = toc_cache["pag_f"]
            gs = toc_cache["gs"]
            pct = 100.0 * float(np.mean(pagf > gs))   # strictly greater
            pagf_gt_gs_pct.append(pct)
        
        if "pag_f" in toc_cache and "pag_nf" in toc_cache:
            pagf = toc_cache["pag_f"]
            pagnf = toc_cache["pag_nf"]
            pct = 100.0 * float(np.mean(pagf > pagnf))  # strictly greater
            pagf_gt_pagnf_pct.append(pct)


    curve_stats: dict[str, dict[str, np.ndarray | float]] = {}
    area_stats: dict[str, dict[str, float]] = {}

    for policy in policies:
        n_pol = count_toc[policy]
        if n_pol == 0:
            if debug:
                print(f"[Aggregate] WARNING: no datasets found for policy={policy}")
            continue

        mean = sum_toc[policy] / n_pol
        # var = E[X^2] - (E[X])^2
        ex2 = sumsq_toc[policy] / n_pol
        var = np.maximum(ex2 - mean * mean, 0.0)
        std = np.sqrt(var)
        sem = std / np.sqrt(n_pol)

        curve_stats[policy] = {"mean": mean, "std": std, "sem": sem, "n": float(n_pol)}

        if policy in AREA_VALID_POLICIES and len(area_vals[policy]) > 0:
            a = np.asarray(area_vals[policy], dtype=float)
            area_mean = float(a.mean())
            area_std = float(a.std(ddof=1)) if a.size >= 2 else 0.0
            area_sem = float(area_std / np.sqrt(a.size)) if a.size >= 2 else 0.0
            area_stats[policy] = {"n": float(a.size), "mean": area_mean, "std": area_std, "sem": area_sem}
    
    if pagf_gt_gs_pct:
        arr = np.asarray(pagf_gt_gs_pct, dtype=float)
        mean_pct = float(arr.mean())
        std_pct = float(arr.std(ddof=1)) if arr.size >= 2 else 0.0
        if debug:
            print(f"\n=== PAG-F vs GSP (strict) ===")
            print(f"% budgets where PAG-F > GSP: {mean_pct:.2f}% ± {std_pct:.2f}% (n={arr.size})")
        
    if pagf_gt_pagnf_pct:
        arr = np.asarray(pagf_gt_pagnf_pct, dtype=float)
        mean_pct = float(arr.mean())
        std_pct = float(arr.std(ddof=1)) if arr.size >= 2 else 0.0
        if debug:
            print(f"\n=== PAG-F vs PAG-NF ({dataset_subdir}) ===")
            print(f"% budgets where PAG-F > PAG-NF: {mean_pct:.2f}% ± {std_pct:.2f}% (n={arr.size})")
    
    # --- plot ---
    plt.figure(figsize=(8, 5))
    x = (np.arange(1, B_common + 1, dtype=float) / B_common) * 100.0

    is_strict = (dataset_subdir == "strictly_increasing")
    is_strict_decreasing = (dataset_subdir == "strictly_decreasing")

    for policy in policies:
        if policy not in curve_stats:
            continue

        m = curve_stats[policy]["mean"]

        # Collapse overlapping policies into ONE line for strictly increasing
        if is_strict and policy in {"gs", "pag_f", "pag_nf"}:
            # Only draw once (use PAG-NF as the representative)
            if policy != "pag_nf":
                continue

            plt.plot(
                x, m,
                label="IFP / PAG* / PAG (overlap)",
                color="black",
                linewidth=2.8,
                zorder=2,
            )
            continue

        # Collapse overlapping policies into ONE line for strictly decreasing
        if is_strict_decreasing and policy in {"pag_nf", "pag_f", "greedy", "prp"}:
            # Only draw once (use PAG-NF as the representative)
            if policy != "pag_nf":
                continue

            plt.plot(
                x, m,
                label="PAG / PAG* / PGP / LFP (overlap)",
                color="black",
                linewidth=2.8,
                linestyle="-",
                zorder=3,
            )
            continue

        color, ls = _style_for_policy(policy)
        plt.plot(
            x, m,
            label=DISPLAY.get(policy, policy),
            color=color,
            linewidth=2.4,
            linestyle=ls,
            zorder=2,
        )

    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Budget used (%)", fontsize=16)
    plt.ylabel("TOC Value", fontsize=16)
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.show()

    # --- print LaTeX summary table ---
    MLAUTOC_print_summary(
        area_stats,
        policies=policies,
        caption=f"ML-AUTOC (area = sum of TOC over budgets 1..{B_common}) across datasets in '{dataset_subdir}'.",
        label=f"tab:mlautoc_{dataset_subdir}",
        show_sem=False,
    )
    if debug:
        print("=== Area (ML-AUTOC) mean ± std ===")
        for p in policies:
            if p in area_stats:
                print(f"{DISPLAY.get(p,p)}: {area_stats[p]['mean']:.6f} ± {area_stats[p]['std']:.6f} (n={int(area_stats[p]['n'])})")

def main():
    # plot_MLAUTOC_randomchoice(dataset_idx="085")
    
    # plotting cumulative graphs for only the strictly increasing dataset
    plot_MLAUTOC_all_datasets(dataset_subdir="strictly_increasing", results_dir="resultsrq1", title_suffix="Strictly Increasing")

    # plotting cumulative graphs for only the non-negative marginal dataset
    plot_MLAUTOC_all_datasets(dataset_subdir="nonnegative", results_dir="resultsrq1nn", title_suffix="Nonnegative")

    # plotting cumulative graphs for only the 
    plot_MLAUTOC_all_datasets(dataset_subdir="strictly_decreasing", results_dir="resultsrq1sd", title_suffix="Strictly Decreasing")
      
if __name__ == "__main__":
    main()

    

