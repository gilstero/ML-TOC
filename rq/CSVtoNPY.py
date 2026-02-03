"""
CSVtoNPY.py is devoted to convert the PAG-F CSV ranking into a NPY format for quicker loading.
Run this script directly.
-----------------------------------------------
Written by Olin Gilster at the TangAI Lab, 2026
-----------------------------------------------
"""

import numpy as np
from pathlib import Path
import re

DELTA_RE = re.compile(r"delta_(\d{3})\.csv$")

"""
Un-comment the dataset and rankings directory you want to convert from CSV to NPY
"""
DATASET_DIR = None
RANKINGS_DIR = None
# DATASET_DIR = Path("datasets/strictly_increasing")
# DATASET_DIR = Path("datasets/nonnegative")
DATASET_DIR = Path("datasets/strictly_decreasing")
# RANKINGS_DIR = Path("resultsrq1/rankings")
# RANKINGS_DIR = Path("resultsrq1nn/rankings")
RANKINGS_DIR = Path("resultsrq1sd/rankings")


ids = sorted([DELTA_RE.search(p.name).group(1) for p in DATASET_DIR.iterdir() if DELTA_RE.search(p.name)])

for idx in ids:
    csv_path = RANKINGS_DIR / f"{idx}_pag_f.csv"
    npy_path = RANKINGS_DIR / f"{idx}_pag_f.npy"
    if npy_path.exists():
        continue

    print("Converting", csv_path)
    arr = np.loadtxt(csv_path, delimiter=",", dtype=np.uint8)
    np.save(npy_path, arr)
