"""
-----------------------------------------------

-----------------------------------------------
"""

import matplotlib.pyplot as plt
from itertools import combinations

# --- Toy data: marginal benefits δ_i^(ℓ) ---
# Items are (id, level) with benefit
items = {
    ("x1", 1): 1.0,
    ("x1", 2): 5.0,
    ("x2", 1): 2.0,
    ("x2", 2): 3.0,
}

def is_precedence_feasible(selection):
    """Precedence: cannot include (i,2) unless (i,1) is included."""
    s = set(selection)
    for (i, l) in s:
        if l == 2 and (i, 1) not in s:
            return False
    return True

def avg_benefit(selection):
    """Average marginal benefit over selected (i,ℓ) pairs (your ATE definition)."""
    return sum(items[p] for p in selection) / len(selection)

def exact_random_baseline(budget):
    """
    RAND_budget: uniform over all precedence-feasible allocations
    that use exactly 'budget' marginal treatments (since cost=1 each).
    """
    all_pairs = list(items.keys())
    feasible_sets = [
        comb for comb in combinations(all_pairs, budget)
        if is_precedence_feasible(comb)
    ]
    if not feasible_sets:
        raise ValueError(f"No feasible allocations of size {budget}")

    return sum(avg_benefit(s) for s in feasible_sets) / len(feasible_sets)

# Budgets (x-axis): number of marginal treatments, since c_ℓ = 1
budgets = [1, 2, 3, 4]

# Your original (raw) ATE curves
oracle_ate = [2.0, 3.0, 8.0 / 3.0, 11.0 / 4.0]
policy_prefix_ate = [2.0, 3.0 / 2.0, 8.0 / 3.0, 11.0 / 4.0]
policy_global_ate = [1.0, 3.0, 8.0 / 3.0, 11.0 / 4.0]
policy_levelwise_ate = [5.0, 4.0, 10.0 / 3.0, 11.0 / 4.0]

# Compute exact random baseline RAND_β for each budget
rand = [exact_random_baseline(b) for b in budgets]

# Convert ATE curves into TOC curves: TOC = ATE - RAND
oracle_toc = [a - r for a, r in zip(oracle_ate, rand)]
policy_prefix_toc = [a - r for a, r in zip(policy_prefix_ate, rand)]
policy_global_toc = [a - r for a, r in zip(policy_global_ate, rand)]
policy_levelwise_toc = [a - r for a, r in zip(policy_levelwise_ate, rand)]

def main():
    plt.figure(figsize=(6, 4))

    plt.axhline(0, linewidth=1)  # zero line: "equal to random"

    plt.plot(
    budgets,
    oracle_toc,
    linewidth=3,
    marker="o",
    markersize=7,
    label="PAG-F",
    zorder=1,
    )

    plt.plot(
        budgets,
        policy_prefix_toc,
        linewidth=1.8,
        linestyle="--",
        marker="s",
        markersize=6,
        label="PRPP",
        zorder=3,  # drawn on top
    )

    plt.plot(
        budgets,
        policy_global_toc,
        linewidth=1.8,
        linestyle="--",
        marker="s",
        markersize=6,
        label="GSP",
        zorder=2,  # drawn on top
    )
    plt.plot(budgets, policy_levelwise_toc, marker="o", label="MEP")

    plt.xlabel("Budget (number of marginal treatments)")
    plt.ylabel("Excess average marginal benefit over random (TOC)")
    plt.title("TOC curves for toy example policies")
    plt.xticks(budgets)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("RAND baseline by budget:")
    for b, r in zip(budgets, rand):
        print(f"  β={b}: RAND={r:.6f}")

if __name__ == "__main__":
    main()
