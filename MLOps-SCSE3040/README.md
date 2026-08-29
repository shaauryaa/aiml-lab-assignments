# MLOps — SCSE3040

Solved lab practicals for **SCSE3040 Machine Learning Operations**, B.Tech CSE
5th semester, Bennett University, session 2026-27.

Submitted by **Shaurya Nigam** — roll number `S24CSEU0497`.

## Course repository

The practicals themselves come from the course repo maintained by the
instructor, **Dr. Gaurav Tripathi** (Assistant Professor, SCSET):

**[Bennett-MLOps-Lab/SCSE3040-Lab](https://github.com/Bennett-MLOps-Lab/SCSE3040-Lab)**

| | |
|---|---|
| All practicals | [README.md](https://github.com/Bennett-MLOps-Lab/SCSE3040-Lab/blob/main/README.md) |
| Environment setup | [SETUP.md](https://github.com/Bennett-MLOps-Lab/SCSE3040-Lab/blob/main/SETUP.md) |
| Pinned course versions | [requirements-lock.txt](https://github.com/Bennett-MLOps-Lab/SCSE3040-Lab/blob/main/requirements-lock.txt) |

This folder holds **my worked solutions and their outputs**. The starter
notebooks, briefs and datasets stay in the course repo above.

## Practicals

| # | Practical | Brief & starter | My solution (run, with outputs) | Artifacts | Self-check |
|---|---|---|---|---|---|
| P01 | Your MLOps Workbench | [brief](https://github.com/Bennett-MLOps-Lab/SCSE3040-Lab/blob/main/P01-workbench/README.md) · [P01.ipynb](https://github.com/Bennett-MLOps-Lab/SCSE3040-Lab/blob/main/P01-workbench/P01.ipynb) | [P01_S24CSEU0497.ipynb](P01_S24CSEU0497.ipynb) | [requirements](P01_requirements.txt) · [my_requirements](P01_my_requirements.txt) | 7 / 7 PASS |
| P02 | Your First Honest Model | [brief](https://github.com/Bennett-MLOps-Lab/SCSE3040-Lab/blob/main/P02-first-model/README.md) · [P02.ipynb](https://github.com/Bennett-MLOps-Lab/SCSE3040-Lab/blob/main/P02-first-model/P02.ipynb) | [P02_S24CSEU0497.ipynb](P02_S24CSEU0497.ipynb) | — | 9 / 9 PASS |

P03–P13 are listed in the [course README](https://github.com/Bennett-MLOps-Lab/SCSE3040-Lab/blob/main/README.md)
but have not been published upstream yet. This table gets a row as each one lands.

## P01 — Your MLOps Workbench

Four habits that make a result reproducible: the right virtual environment,
pinned library versions, a fixed random seed, and a git commit.

- **T1** — rebuilt the delivery distances with seed `7` instead of `42` → `[7.69, 10.82, 9.42]`
- **T2** — wrote `my_requirements.txt`, versions asked from Python rather than typed by hand
- **T3** — `fingerprint(path)` returning `rows` / `sha256` / `seed` → 600 rows, seed 42, sha `9e9f7a46…`

All 42 cells are committed with their outputs visible, so the self-check table
and the `git log` output render directly on GitHub.

`P01_requirements.txt` is the four-library file written by Step 3 of the
walkthrough; `P01_my_requirements.txt` is the three-library file written by
Task T2. Both are named per-practical so P02 onwards can sit beside them in
this flat folder.

## P02 --- Your First Honest Model

Split the data, train a model, and prove it beats guessing. All predictors scored
on the same 120 held-out orders (80/20 split, seed 42).

| Predictor | Test-set MAE |
|---|---|
| Mean baseline | 10.32 min |
| Median baseline | 10.33 min |
| **Linear regression** | **1.92 min** |

The model beats the mean baseline by 8.40 minutes, or 81%. RMSE is 2.48, close
enough to MAE that no large misses are hiding among the good predictions.

- **T1** --- median baseline MAE 10.33, a hair worse than the mean's 10.32, so `"mean"` wins
- **T2** --- re-splitting 70/30 with seed 7 gave MAE 2.10 instead of 1.92; the conclusion held, and that 0.18-minute wobble is why a single split is an approximate figure rather than an exact one
- **T3** --- `predict_minutes()` returns one number per order; the rainy 3 km order is 35.0 min against 29.5 min dry, exactly the 5.5-minute rain coefficient

Learned coefficients (3.07 min/km, 4.13 per traffic level, 5.55 for rain, 0.65 per
prep minute) sit within rounding of the formula that generated the dataset.

## Environment note

My local venv drifted from the course lock file. Recorded here because a
practical about reproducibility is the wrong place to leave this undocumented:

| Package | Course lock file | My environment |
|---|---|---|
| numpy | `2.5.1` | `2.4.6` |
| pandas | `2.3.3` | `3.0.5` |
| scikit-learn | `1.9.0` | `1.9.0` |
| matplotlib | `3.11.1` | `3.11.1` |

`pandas` is a full major version ahead of the pin. Every P01 self-check still
passes, but later practicals should be run after rebuilding the venv from
[requirements-lock.txt](https://github.com/Bennett-MLOps-Lab/SCSE3040-Lab/blob/main/requirements-lock.txt).

## Running a practical

```
cd P01-workbench
..\.venv\Scripts\python -m jupyter lab P01.ipynb
```

Work down the notebook with **Shift + Enter**, in order from the top. If it
goes wrong: **Kernel → Restart Kernel and Clear All Outputs**, then start again.
