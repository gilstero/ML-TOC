# ML-AUTOC Experiments

This repository contains code to reproduce the experiments for evaluating ranking policies under multi-level treatment settings.

---

## Project Structure

Navigate to the main directory (`experiment/`). You should see:

* `checkflips.py` – analyzes flip behavior in PAG-F rankings
* `nonnegativemarginals.py` – generates datasets with nonnegative marginals
* `strictlyincreasing.py` – generates strictly increasing marginal datasets
* `strictlydecreasing.py` – generates strictly decreasing datasets
* `runexperiments.py` – main experiment runner
* `toyexamplegraph.py` – reproduces Figure 1 from the paper
* `instructions.txt` – original instructions

### `rq/` directory:

* `CSVtoNPY.py` – converts dataset formats
* `rq1.py` – runs policies and generates rankings
* `rq1andrq2solutions.py` – computes evaluation metrics (ML-AUTOC, TOC, win rates)
* `timing.py` – benchmarks runtime of policies

Each file includes additional details at the top describing its purpose.

---

## Running an Experiment

Follow these steps to reproduce results:

### 1. Setup

Download all files and ensure required packages are installed (see below).

---

### 2. Generate Datasets

Run:

```python
python createdatasets.py
```

This generates:

* strictly increasing datasets
* nonnegative marginal datasets

---

### 3. Run Ranking Policies

Navigate to the `rq/` directory and run:

```python
python rq1.py
```

This script:

* applies all policies
* generates rankings for each dataset

See comments inside `rq1.py` for configuration options.

---

### 4. Evaluate Results

Run:

```python
python rq1andrq2solutions.py
```

This outputs:

* ML-AUTOC values
* TOC curves
* Percentage win rates of PAG-F

---

## Dependencies

These experiments were run using Python 3.9.7.

Required packages:

* numpy
* matplotlib
* tqdm

Standard library modules used:

* re
* pathlib
* os
* sys
* itertools

---

## Additional Utilities

* `toyexamplegraph.py`
  Recreates the toy example figure from the paper (Figure 1)

* `checkflips.py`
  Measures how often rankings change ("flips") under PAG-F

---

## Notes

* Avoid re-running dataset generation if files already exist to prevent overwriting.
* Recommended limits:

  * number of levels: ( L < 4 )
  * number of samples: ( n < 1000 )
to avoid long runtimes and large storage usage.
