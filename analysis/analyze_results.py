#!/usr/bin/env python3
"""
Statistical analysis of the CoCo vs CE / SupCon / Dot-Regression experiments
on OpenML-CC18 (the analysis reported in the paper).

Reads the results folder (one sub-folder task<ID>_<name>/ per dataset with
outer_metrics.csv, dataset_info.json, best_params.csv, config.json, histories/)
and writes to --out:

  report.md                 Markdown report with all results
  results_table.tex         LaTeX table: mean ± std and mean Friedman rank
  per_dataset_means.csv     mean over the 10 outer folds, per dataset and method
  per_dataset_wtl.csv       CoCo win/tie/loss per dataset (Wilcoxon + Holm)
  across_dataset_tests.csv  Friedman + Wilcoxon-Holm on the per-dataset means
  imbalance.csv             imbalance ratio and CoCo gain per dataset
  training_dynamics.csv     epochs, run time and validation score after epoch 1

Usage:
  python analysis/analyze_results.py --results results --out results/analysis

Requirements: pandas, numpy, scipy.
"""
import argparse
import glob
import json
import os
import re
import warnings

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, spearmanr, wilcoxon

warnings.filterwarnings("ignore")

METHODS = ["coco", "ce", "supcon", "dr"]
BASELINES = ["ce", "supcon", "dr"]
NAMES = {"coco": "CoCo", "ce": "CE", "supcon": "SupCon", "dr": "Dot-Reg."}
METRICS = ["balanced_accuracy", "f1_macro", "cohen_kappa", "precision_macro"]
METRIC_NAMES = {"balanced_accuracy": "Bal. Acc.", "f1_macro": "F1 macro",
                "cohen_kappa": "Cohen's kappa", "precision_macro": "Prec. macro"}
IR_THRESHOLDS = [1.5, 2, 3, 5, 10]


# ----------------------------------------------------------------------------
# Statistical utilities
# ----------------------------------------------------------------------------
def holm(pvals):
    """Holm correction. Returns adjusted p-values in the original order."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    k = len(p)
    adj = np.empty(k)
    running = 0.0
    for i, j in enumerate(order):
        running = max(running, (k - i) * p[j])
        adj[j] = min(1.0, running)
    return adj


def safe_wilcoxon(a, b):
    """Two-sided paired Wilcoxon signed-rank test; p = 1 if all differences are zero."""
    d = np.asarray(a) - np.asarray(b)
    if np.allclose(d, 0):
        return 1.0
    try:
        return float(wilcoxon(a, b).pvalue)
    except ValueError:
        return 1.0


def fmt_p(p):
    return f"{p:.2g}" if p >= 0.001 else f"{p:.1e}"


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def task_id(folder):
    m = re.match(r"task(\d+)_", os.path.basename(folder.rstrip("/")))
    return int(m.group(1)) if m else 10**9


def load(results):
    """Load the finished tasks (those with a DONE file) and drop duplicated
    datasets (the lowest OpenML task id is kept)."""
    folders = [f for f in glob.glob(os.path.join(results, "task*/"))
               if os.path.exists(os.path.join(f, "DONE"))]
    folders.sort(key=task_id)

    outer, info, best, configs = [], {}, [], []
    seen = {}
    duplicates = []
    for f in folders:
        di = json.load(open(os.path.join(f, "dataset_info.json")))
        name = di["name"]
        if name in seen:
            duplicates.append((name, os.path.basename(seen[name].rstrip("/")),
                               os.path.basename(f.rstrip("/"))))
            continue
        seen[name] = f
        m = pd.read_csv(os.path.join(f, "outer_metrics.csv"))
        m["task"] = os.path.basename(f.rstrip("/"))
        outer.append(m)
        counts = np.array(list(di["class_counts"].values()), dtype=float)
        info[name] = dict(task=m["task"].iloc[0], n=di["n"], p=di["d_encoded"],
                          C=di["n_classes"], IR=counts.max() / counts.min())
        bp = os.path.join(f, "best_params.csv")
        if os.path.exists(bp):
            best.append(pd.read_csv(bp))
        cfg = json.load(open(os.path.join(f, "config.json")))
        configs.append(dict(dataset=name, **cfg.get("train", {})))

    if not outer:
        raise SystemExit(f"No finished tasks (task*/DONE) found in {results}")
    outer = pd.concat(outer, ignore_index=True)
    info = pd.DataFrame(info).T
    for c in ["n", "p", "C", "IR"]:
        info[c] = info[c].astype(float)
    best = pd.concat(best, ignore_index=True) if best else pd.DataFrame()
    configs = pd.DataFrame(configs)

    unfinished = sorted(os.path.basename(f.rstrip("/"))
                        for f in glob.glob(os.path.join(results, "task*/"))
                        if not os.path.exists(os.path.join(f, "DONE")))
    return outer, info, best, configs, duplicates, unfinished, seen


# ----------------------------------------------------------------------------
# Analyses
# ----------------------------------------------------------------------------
def summary_table(outer):
    rows = []
    for met in METRICS:
        means = outer.pivot_table(index="dataset", columns="method", values=met)
        ranks = means[METHODS].rank(axis=1, ascending=False).mean()
        for m in METHODS:
            rows.append(dict(metric=met, method=m, mean=means[m].mean(),
                             std=means[m].std(), friedman_rank=ranks[m]))
    return pd.DataFrame(rows)


def per_dataset_wtl(outer, alpha):
    """Per dataset: paired Wilcoxon over the outer folds, CoCo vs each baseline,
    Holm over the 3 CoCo comparisons."""
    rows = []
    for met in METRICS:
        P = outer.pivot_table(index=["dataset", "outer_fold"], columns="method", values=met)
        for ds, g in P.groupby(level="dataset"):
            ps = [safe_wilcoxon(g["coco"], g[o]) for o in BASELINES]
            adj = holm(ps)
            for o, p, pa in zip(BASELINES, ps, adj):
                diff = (g["coco"] - g[o]).mean()
                res = "tie"
                if pa < alpha:
                    res = "win" if diff > 0 else "loss"
                rows.append(dict(metric=met, dataset=ds, baseline=o, mean_diff=diff,
                                 p=p, p_holm=pa, result=res, n_folds=len(g)))
    return pd.DataFrame(rows)


def across_datasets(outer):
    """Friedman (4 methods) + Wilcoxon CoCo vs each baseline on the per-dataset
    means, with Holm over the 3 comparisons."""
    rows = []
    for met in METRICS:
        means = outer.pivot_table(index="dataset", columns="method", values=met)
        fr = friedmanchisquare(*[means[m] for m in METHODS])
        ps = [safe_wilcoxon(means["coco"], means[o]) for o in BASELINES]
        adj = holm(ps)
        for o, p, pa in zip(BASELINES, ps, adj):
            d = means["coco"] - means[o]
            rows.append(dict(metric=met, friedman_chi2=fr.statistic, friedman_p=fr.pvalue,
                             baseline=o, coco_better=(d > 0).sum(), coco_worse=(d < 0).sum(),
                             mean_diff=d.mean(), wilcoxon_p=p, wilcoxon_p_holm=pa))
    return pd.DataFrame(rows)


def gap_to_best(outer, met="balanced_accuracy"):
    means = outer.pivot_table(index="dataset", columns="method", values=met)[METHODS]
    best = means.max(axis=1)
    gap = means.rsub(best, axis=0)
    winners = means.idxmax(axis=1).value_counts()
    return gap.mean(), gap.max(), winners


def imbalance(outer, info, met="balanced_accuracy"):
    means = outer.pivot_table(index="dataset", columns="method", values=met)
    df = means.join(info[["n", "p", "C", "IR"]])
    corr, subsets = [], []
    for o in BASELINES:
        r = spearmanr(np.log(df["IR"]), df["coco"] - df[o])
        corr.append(dict(baseline=o, spearman_rho=r.correlation, p=r.pvalue))
    for th in IR_THRESHOLDS:
        s = df[df["IR"] >= th]
        for o in BASELINES:
            d = s["coco"] - s[o]
            subsets.append(dict(IR_min=th, n_datasets=len(s), baseline=o,
                                coco_better=(d > 0).sum(), coco_worse=(d < 0).sum(),
                                gain_points=100 * d.mean(),
                                wilcoxon_p=safe_wilcoxon(s["coco"], s[o]) if len(s) > 1 else np.nan,
                                coco_mean_rank=s[METHODS].rank(axis=1, ascending=False)["coco"].mean()))
    return df, pd.DataFrame(corr), pd.DataFrame(subsets)


def training_dynamics(outer, results, seen):
    """Epochs until early stopping, time per epoch and validation score after
    the first epoch (from histories/)."""
    o = outer.copy()
    o["time_per_epoch"] = o["final_train_time_s"] / o["epochs_run"]
    per_ds = o.groupby(["dataset", "method"])[["epochs_run", "best_epoch",
                                                "final_train_time_s", "time_per_epoch"]].mean()
    ep1 = []
    for ds, folder in seen.items():
        for m in METHODS:
            vals = []
            for fn in glob.glob(os.path.join(folder, "histories", f"{m}_fold*.csv")):
                h = pd.read_csv(fn)
                if len(h):
                    vals.append(h["es_score"].iloc[0])
            if vals:
                ep1.append(dict(dataset=ds, method=m, val_score_epoch1=np.mean(vals)))
    ep1 = pd.DataFrame(ep1).set_index(["dataset", "method"]) if ep1 else None
    if ep1 is not None:
        per_ds = per_ds.join(ep1)
    per_ds = per_ds.reset_index()

    comps = []
    cols = ["epochs_run", "best_epoch", "time_per_epoch"] + (["val_score_epoch1"] if ep1 is not None else [])
    for col in cols:
        X = per_ds.pivot(index="dataset", columns="method", values=col)
        for a, b in [("coco", "ce"), ("coco", "supcon"), ("coco", "dr"),
                     ("supcon", "ce"), ("dr", "ce")]:
            d = X[a] - X[b]
            comps.append(dict(variable=col, a=a, b=b, mean_a=X[a].mean(), mean_b=X[b].mean(),
                              a_higher=(d > 0).sum(), a_lower=(d < 0).sum(),
                              median_ratio=(X[a] / X[b]).median(),
                              wilcoxon_p=safe_wilcoxon(X[a], X[b])))
    return per_ds, pd.DataFrame(comps)


# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------
def write_latex_table(summ, path):
    lines = [r"\begin{tabular}{lcccc}", r"\toprule",
             "Metric & " + " & ".join(NAMES[m] for m in METHODS) + r" \\", r"\midrule"]
    for met in METRICS:
        s = summ[summ.metric == met].set_index("method")
        best_mean = s["mean"].idxmax()
        cells = []
        for m in METHODS:
            mean = f"{s.loc[m, 'mean']:.3f}".lstrip("0")
            std = f"{s.loc[m, 'std']:.3f}".lstrip("0")
            cell = f"{mean}{{\\scriptsize$\\pm${std}}} ({s.loc[m, 'friedman_rank']:.2f})"
            if m == best_mean:
                cell = r"\textbf{" + cell + "}"
            cells.append(cell)
        lines.append(METRIC_NAMES[met] + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, help="results/ folder containing the task* sub-folders")
    ap.add_argument("--out", default=None, help="output folder (default: <results>/analysis)")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args(argv)
    out = args.out or os.path.join(args.results, "analysis")
    os.makedirs(out, exist_ok=True)

    outer, info, best, configs, duplicates, unfinished, seen = load(args.results)
    n_ds = outer["dataset"].nunique()
    n_folds = outer.groupby(["dataset", "method"]).size().max()

    summ = summary_table(outer)
    wtl = per_dataset_wtl(outer, args.alpha)
    acr = across_datasets(outer)
    gap_mean, gap_max, winners = gap_to_best(outer)
    imb_df, imb_corr, imb_sub = imbalance(outer, info)
    dyn, dyn_comp = training_dynamics(outer, args.results, seen)

    # CSV files
    outer.pivot_table(index="dataset", columns="method", values=METRICS).to_csv(
        os.path.join(out, "per_dataset_means.csv"))
    wtl.to_csv(os.path.join(out, "per_dataset_wtl.csv"), index=False)
    acr.to_csv(os.path.join(out, "across_dataset_tests.csv"), index=False)
    imb_df.to_csv(os.path.join(out, "imbalance.csv"))
    dyn.to_csv(os.path.join(out, "training_dynamics.csv"), index=False)
    write_latex_table(summ, os.path.join(out, "results_table.tex"))

    # Markdown report
    L = []
    w = L.append
    w("# Statistical analysis: CoCo vs CE / SupCon / Dot-Regression\n")
    w(f"Results folder: `{os.path.normpath(args.results)}`\n")
    w("## 0. Data\n")
    w(f"- Datasets analysed: **{n_ds}**, with up to {n_folds} outer folds per method.")
    if duplicates:
        w("- Duplicated datasets (the lowest OpenML task id is used): " +
          "; ".join(f"{d} ({k} used, {s} discarded)" for d, k, s in duplicates) + ".")
    if unfinished:
        w(f"- Unfinished tasks (no DONE file), excluded: {', '.join(unfinished)}.")
    if "supcon_temperature" in configs:
        temps = configs["supcon_temperature"].unique()
        w(f"- SupCon temperature in config.json: {', '.join(map(str, temps))}.")
    if len(best) and "best_coco_norm" in best:
        c = best[(best.method == "coco") & best.dataset.isin(seen.keys())]
        k = c["best_coco_norm"].astype(str).str.lower().eq("true").sum()
        w(f"- L2 normalisation of CoCo selected in {k}/{len(c)} outer folds "
          f"({100 * k / len(c):.0f}%).")
    w("")

    w("## 1. Summary (mean ± std over datasets, mean Friedman rank)\n")
    w("| Metric | " + " | ".join(NAMES[m] for m in METHODS) + " |")
    w("|---|" + "---|" * len(METHODS))
    for met in METRICS:
        s = summ[summ.metric == met].set_index("method")
        w(f"| {METRIC_NAMES[met]} | " + " | ".join(
            f"{s.loc[m, 'mean']:.4f} ± {s.loc[m, 'std']:.4f} ({s.loc[m, 'friedman_rank']:.2f})"
            for m in METHODS) + " |")
    w("\nFriedman rank: 1 = best; computed on the per-dataset means.\n")

    w("## 2. Tests across datasets (unit = dataset)\n")
    w("Friedman test with the 4 methods on the per-dataset means, followed by paired "
      "Wilcoxon tests CoCo vs each baseline with Holm correction over the 3 comparisons "
      "(Demšar, 2006).\n")
    w("| Metric | Friedman p | Baseline | CoCo better / worse | Mean diff. | Wilcoxon p | Holm p |")
    w("|---|---|---|---|---|---|---|")
    for _, r in acr.iterrows():
        sig = " **" if r.wilcoxon_p_holm < args.alpha and r.friedman_p < args.alpha else ""
        w(f"| {METRIC_NAMES[r.metric]} | {fmt_p(r.friedman_p)} | {NAMES[r.baseline]} | "
          f"{r.coco_better} / {r.coco_worse} | {100 * r.mean_diff:+.2f} pt | "
          f"{fmt_p(r.wilcoxon_p)} | {fmt_p(r.wilcoxon_p_holm)}{sig} |")
    w("\n** = significant under both Friedman and Wilcoxon-Holm. If Friedman is not "
      "significant, the Wilcoxon tests for that metric should be read as exploratory.\n")

    w("## 3. Tests within each dataset (unit = outer fold)\n")
    w("For each dataset, paired Wilcoxon test over the outer folds (same partitions for "
      "all methods), CoCo vs each baseline, Holm over the 3 comparisons. "
      "Win/loss = significant in favour of/against CoCo. Caveat: folds are not "
      "independent (training sets overlap), so these tests are somewhat optimistic.\n")
    w("| Metric | vs CE (W/T/L) | vs SupCon (W/T/L) | vs DR (W/T/L) |")
    w("|---|---|---|---|")
    for met in METRICS:
        cells = []
        for o in BASELINES:
            s = wtl[(wtl.metric == met) & (wtl.baseline == o)].result.value_counts()
            cells.append(f"{s.get('win', 0)}/{s.get('tie', 0)}/{s.get('loss', 0)}")
        w(f"| {METRIC_NAMES[met]} | " + " | ".join(cells) + " |")
    w("")

    w("## 4. Gap to the best method (balanced accuracy)\n")
    w("Per dataset: mean of the best method minus mean of each method.\n")
    w("| Method | Mean gap (pt) | Max gap (pt) | Best mean on n datasets |")
    w("|---|---|---|---|")
    for m in METHODS:
        w(f"| {NAMES[m]} | {100 * gap_mean[m]:.2f} | {100 * gap_max[m]:.2f} | {winners.get(m, 0)} |")
    w("")

    w("## 5. Class imbalance (balanced accuracy, exploratory analysis)\n")
    w("Imbalance ratio = samples in the largest class / smallest class (after preprocessing).\n")
    w("Spearman correlation between log(ratio) and the per-dataset gain of CoCo:\n")
    w("| Baseline | rho | p |")
    w("|---|---|---|")
    for _, r in imb_corr.iterrows():
        w(f"| {NAMES[r.baseline]} | {r.spearman_rho:.2f} | {fmt_p(r.p)} |")
    w("\nSubsets by imbalance-ratio threshold (Wilcoxon on per-dataset means, uncorrected):\n")
    w("| Ratio ≥ | n datasets | Baseline | CoCo better / worse | Gain (pt) | Wilcoxon p | CoCo mean rank |")
    w("|---|---|---|---|---|---|---|")
    for _, r in imb_sub.iterrows():
        w(f"| {r.IR_min} | {r.n_datasets} | {NAMES[r.baseline]} | {r.coco_better} / {r.coco_worse} | "
          f"{r.gain_points:+.2f} | {fmt_p(r.wilcoxon_p)} | {r.coco_mean_rank:.2f} |")
    w("")

    w("## 6. Training dynamics and cost\n")
    w("Per-dataset means of: epochs until early stopping, best epoch, time per epoch "
      "(final training run) and validation score after the first epoch. "
      "Note: CoCo processes two batches per step.\n")
    w("| Variable | A vs B | Mean A | Mean B | A higher / lower | Median ratio A/B | Wilcoxon p |")
    w("|---|---|---|---|---|---|---|")
    for _, r in dyn_comp.iterrows():
        w(f"| {r.variable} | {NAMES[r.a]} vs {NAMES[r.b]} | {r.mean_a:.4g} | {r.mean_b:.4g} | "
          f"{r.a_higher} / {r.a_lower} | {r.median_ratio:.2f} | {fmt_p(r.wilcoxon_p)} |")
    w("")

    with open(os.path.join(out, "report.md"), "w") as f:
        f.write("\n".join(L))
    print(f"Report and tables written to: {out}")


if __name__ == "__main__":
    main()
