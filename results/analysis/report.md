# Statistical analysis: CoCo vs CE / SupCon / Dot-Regression

Results folder: `results`

## 0. Data

- Datasets analysed: **72**, with up to 10 outer folds per method.
- Duplicated datasets (the lowest OpenML task id is used): kr-vs-kp (task3_kr-vs-kp used, task63_kr-vs-kp discarded); arrhythmia (task5_arrhythmia used, task65_arrhythmia discarded); balance-scale (task11_balance-scale used, task70_balance-scale discarded); mfeat-factors (task12_mfeat-factors used, task71_mfeat-factors discarded).
- SupCon temperature in config.json: 0.07.
- L2 normalisation of CoCo selected in 506/720 outer folds (70%).

## 1. Summary (mean ± std over datasets, mean Friedman rank)

| Metric | CoCo | CE | SupCon | Dot-Reg. |
|---|---|---|---|---|
| Bal. Acc. | 0.8063 ± 0.1782 (2.17) | 0.7904 ± 0.1933 (2.72) | 0.8022 ± 0.1877 (2.45) | 0.7958 ± 0.1894 (2.65) |
| F1 macro | 0.7965 ± 0.1867 (2.35) | 0.7829 ± 0.2025 (2.68) | 0.7886 ± 0.1973 (2.46) | 0.7901 ± 0.1969 (2.51) |
| Cohen's kappa | 0.6789 ± 0.2873 (2.20) | 0.6606 ± 0.3128 (2.68) | 0.6608 ± 0.3128 (2.56) | 0.6640 ± 0.3041 (2.56) |
| Prec. macro | 0.8023 ± 0.1856 (2.51) | 0.7955 ± 0.1949 (2.47) | 0.7905 ± 0.1957 (2.62) | 0.8000 ± 0.1906 (2.39) |

Friedman rank: 1 = best; computed on the per-dataset means.

## 2. Tests across datasets (unit = dataset)

Friedman test with the 4 methods on the per-dataset means, followed by paired Wilcoxon tests CoCo vs each baseline with Holm correction over the 3 comparisons (Demšar, 2006).

| Metric | Friedman p | Baseline | CoCo better / worse | Mean diff. | Wilcoxon p | Holm p |
|---|---|---|---|---|---|---|
| Bal. Acc. | 0.049 | CE | 48 / 24 | +1.60 pt | 0.0062 | 0.012 ** |
| Bal. Acc. | 0.049 | SupCon | 41 / 30 | +0.42 pt | 0.15 | 0.15 |
| Bal. Acc. | 0.049 | Dot-Reg. | 42 / 30 | +1.06 pt | 0.0033 | 0.01 ** |
| F1 macro | 0.48 | CE | 45 / 27 | +1.36 pt | 0.11 | 0.33 |
| F1 macro | 0.48 | SupCon | 37 / 35 | +0.79 pt | 0.31 | 0.33 |
| F1 macro | 0.48 | Dot-Reg. | 37 / 35 | +0.64 pt | 0.15 | 0.33 |
| Cohen's kappa | 0.13 | CE | 47 / 25 | +1.83 pt | 0.091 | 0.098 |
| Cohen's kappa | 0.13 | SupCon | 43 / 28 | +1.81 pt | 0.011 | 0.033 |
| Cohen's kappa | 0.13 | Dot-Reg. | 39 / 33 | +1.49 pt | 0.049 | 0.098 |
| Prec. macro | 0.74 | CE | 37 / 35 | +0.68 pt | 0.61 | 1 |
| Prec. macro | 0.74 | SupCon | 38 / 34 | +1.18 pt | 0.18 | 0.54 |
| Prec. macro | 0.74 | Dot-Reg. | 32 / 40 | +0.22 pt | 0.82 | 1 |

** = significant under both Friedman and Wilcoxon-Holm. If Friedman is not significant, the Wilcoxon tests for that metric should be read as exploratory.

## 3. Tests within each dataset (unit = outer fold)

For each dataset, paired Wilcoxon test over the outer folds (same partitions for all methods), CoCo vs each baseline, Holm over the 3 comparisons. Win/loss = significant in favour of/against CoCo. Caveat: folds are not independent (training sets overlap), so these tests are somewhat optimistic.

| Metric | vs CE (W/T/L) | vs SupCon (W/T/L) | vs DR (W/T/L) |
|---|---|---|---|
| Bal. Acc. | 14/54/4 | 7/63/2 | 10/59/3 |
| F1 macro | 13/53/6 | 6/64/2 | 8/61/3 |
| Cohen's kappa | 14/53/5 | 6/64/2 | 8/61/3 |
| Prec. macro | 9/54/9 | 5/66/1 | 6/59/7 |

## 4. Gap to the best method (balanced accuracy)

Per dataset: mean of the best method minus mean of each method.

| Method | Mean gap (pt) | Max gap (pt) | Best mean on n datasets |
|---|---|---|---|
| CoCo | 1.36 | 19.56 | 25 |
| CE | 2.96 | 29.48 | 18 |
| SupCon | 1.78 | 22.50 | 17 |
| Dot-Reg. | 2.42 | 16.00 | 12 |

## 5. Class imbalance (balanced accuracy, exploratory analysis)

Imbalance ratio = samples in the largest class / smallest class (after preprocessing).

Spearman correlation between log(ratio) and the per-dataset gain of CoCo:

| Baseline | rho | p |
|---|---|---|
| CE | 0.32 | 0.0056 |
| SupCon | 0.14 | 0.23 |
| Dot-Reg. | 0.37 | 0.0012 |

Subsets by imbalance-ratio threshold (Wilcoxon on per-dataset means, uncorrected):

| Ratio ≥ | n datasets | Baseline | CoCo better / worse | Gain (pt) | Wilcoxon p | CoCo mean rank |
|---|---|---|---|---|---|---|
| 1.5 | 42 | CE | 28 / 14 | +2.44 | 0.0025 | 1.98 |
| 1.5 | 42 | SupCon | 27 / 15 | +0.14 | 0.24 | 1.98 |
| 1.5 | 42 | Dot-Reg. | 30 / 12 | +1.51 | 0.0013 | 1.98 |
| 2.0 | 33 | CE | 24 / 9 | +3.26 | 5.6e-04 | 1.82 |
| 2.0 | 33 | SupCon | 23 / 10 | +0.22 | 0.18 | 1.82 |
| 2.0 | 33 | Dot-Reg. | 25 / 8 | +1.81 | 0.0016 | 1.82 |
| 3.0 | 24 | CE | 19 / 5 | +3.88 | 0.0012 | 1.75 |
| 3.0 | 24 | SupCon | 16 / 8 | -0.25 | 0.88 | 1.75 |
| 3.0 | 24 | Dot-Reg. | 19 / 5 | +2.03 | 0.0059 | 1.75 |
| 5.0 | 19 | CE | 14 / 5 | +3.48 | 0.018 | 1.84 |
| 5.0 | 19 | SupCon | 12 / 7 | -0.20 | 0.98 | 1.84 |
| 5.0 | 19 | Dot-Reg. | 15 / 4 | +1.69 | 0.011 | 1.84 |
| 10.0 | 8 | CE | 5 / 3 | +1.87 | 0.38 | 2.25 |
| 10.0 | 8 | SupCon | 3 / 5 | -1.53 | 0.15 | 2.25 |
| 10.0 | 8 | Dot-Reg. | 6 / 2 | +0.08 | 0.38 | 2.25 |

## 6. Training dynamics and cost

Per-dataset means of: epochs until early stopping, best epoch, time per epoch (final training run) and validation score after the first epoch. Note: CoCo processes two batches per step.

| Variable | A vs B | Mean A | Mean B | A higher / lower | Median ratio A/B | Wilcoxon p |
|---|---|---|---|---|---|---|
| epochs_run | CoCo vs CE | 13.15 | 12.36 | 40 / 32 | 1.06 | 0.059 |
| epochs_run | CoCo vs SupCon | 13.15 | 12.53 | 42 / 26 | 1.04 | 0.082 |
| epochs_run | CoCo vs Dot-Reg. | 13.15 | 12.49 | 36 / 35 | 1.01 | 0.28 |
| epochs_run | SupCon vs CE | 12.53 | 12.36 | 33 / 39 | 0.99 | 0.53 |
| epochs_run | Dot-Reg. vs CE | 12.49 | 12.36 | 30 / 37 | 0.99 | 0.98 |
| best_epoch | CoCo vs CE | 8.153 | 7.356 | 40 / 32 | 1.09 | 0.056 |
| best_epoch | CoCo vs SupCon | 8.153 | 7.529 | 42 / 26 | 1.07 | 0.079 |
| best_epoch | CoCo vs Dot-Reg. | 8.153 | 7.488 | 36 / 35 | 1.02 | 0.29 |
| best_epoch | SupCon vs CE | 7.529 | 7.356 | 33 / 39 | 0.98 | 0.53 |
| best_epoch | Dot-Reg. vs CE | 7.488 | 7.356 | 30 / 37 | 0.98 | 0.97 |
| time_per_epoch | CoCo vs CE | 0.2037 | 0.1126 | 72 / 0 | 1.66 | 1.7e-13 |
| time_per_epoch | CoCo vs SupCon | 0.2037 | 0.1664 | 71 / 1 | 1.16 | 8.0e-13 |
| time_per_epoch | CoCo vs Dot-Reg. | 0.2037 | 0.137 | 72 / 0 | 1.40 | 1.7e-13 |
| time_per_epoch | SupCon vs CE | 0.1664 | 0.1126 | 72 / 0 | 1.45 | 1.7e-13 |
| time_per_epoch | Dot-Reg. vs CE | 0.137 | 0.1126 | 71 / 1 | 1.20 | 1.7e-13 |
| val_score_epoch1 | CoCo vs CE | 0.6645 | 0.6653 | 34 / 38 | 0.98 | 0.72 |
| val_score_epoch1 | CoCo vs SupCon | 0.6645 | 0.7207 | 25 / 46 | 0.96 | 7.3e-05 |
| val_score_epoch1 | CoCo vs Dot-Reg. | 0.6645 | 0.6806 | 37 / 35 | 1.00 | 0.66 |
| val_score_epoch1 | SupCon vs CE | 0.7207 | 0.6653 | 49 / 23 | 1.03 | 8.4e-04 |
| val_score_epoch1 | Dot-Reg. vs CE | 0.6806 | 0.6653 | 46 / 25 | 1.01 | 0.0056 |
