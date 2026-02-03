"""
Below is the code to check for flips in the allocations produced by PAG-F
The code will print out a summary of any flips found, along with some examples.
You can change the file path in the load_pag_f_csv call to check different allocation files.
-----------------------------------------------
Written by Olin Gilster at the TangAI Lab, 2026
-----------------------------------------------
"""

import numpy as np
from pathlib import Path

def check_flips_from_allocs(allocs_by_budget: np.ndarray, n: int, L: int, max_show: int = 5) -> dict:
    """
    allocs_by_budget: (B, n*L) array of {0,1} allocations, row b is allocation at budget b+1.
    A flip occurs if any 1 in row (b-1) becomes 0 in row b.

    Returns a dict with summary stats and some example flips.
    """
    B, K = allocs_by_budget.shape
    assert K == n * L, f"Expected n*L={n*L}, got {K}"

    flips = []  # list of tuples: (b, num_drops, dropped_indices, added_indices)
    total_drops = 0

    for b in range(1, B):  # compare row b-1 (budget b) to row b (budget b+1)
        prev = allocs_by_budget[b - 1].astype(np.int8)
        curr = allocs_by_budget[b].astype(np.int8)

        diff = curr - prev
        dropped = np.where(diff == -1)[0]
        added   = np.where(diff ==  1)[0]

        if dropped.size > 0:
            total_drops += dropped.size
            flips.append((b + 1, int(dropped.size), dropped, added))  # budget is b+1 in 1-indexed budgets

    # print
    if flips:
        print(f"FOUND FLIPS: {len(flips)} budget transitions had drops.")
        print(f"Total dropped increments across all transitions: {total_drops}")
        print("\nExamples:")
        for j, (budget, num_drops, dropped, added) in enumerate(flips[:max_show], start=1):
            print(f"\nExample {j}: transition to budget {budget}")
            print(f"  drops: {num_drops}, adds: {len(added)}")
            # show up to 10 dropped/added increments as (i,l)
            show_d = dropped[:10]
            show_a = added[:10]
            print("  dropped (i,l):", [(int(idx // L), int(idx % L)) for idx in show_d])
            print("  added   (i,l):", [(int(idx // L), int(idx % L)) for idx in show_a])
    else:
        print("NO FLIPS detected: allocations appear nested across all budgets.")

    return {
        "has_flips": bool(flips),
        "num_flip_transitions": len(flips),
        "total_drops": total_drops,
        "flip_transitions": flips,  # careful: can be large
    }

def load_pag_f_csv(path):
    return np.loadtxt(path, delimiter=",", dtype=np.uint8)

n = 1000
L = 4

# change this file path to check other allocation files:
# ex: allocs = load_pag_f_csv("resultsrq1/rankings/076_pag_f.csv")
allocs = load_pag_f_csv("resultsrq1nn/rankings/085_pag_f.csv")

check_flips_from_allocs(allocs, n=n, L=L, max_show=3)