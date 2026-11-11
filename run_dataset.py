#!/usr/bin/env python
"""Run the full nested-CV benchmark (CE, SupCon, CoCo, DR) on ONE dataset.

Examples
  python run_dataset.py --task-id 31                    # OpenML task (CC18 uses task ids)
  python run_dataset.py --sklearn iris --quick          # bundled dataset, tiny smoke run
  python run_dataset.py --csv data.csv --target label
"""
import argparse
import os
import re
import sys
import time

import torch

from cocobench.data import load_openml_task, load_sklearn, load_csv
from cocobench.nested_cv import run_nested_cv, DEFAULT_LRS, DEFAULT_WDS
from cocobench.train import TrainConfig, METHODS, resolve_device


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--task-id", type=int, help="OpenML task id")
    src.add_argument("--sklearn", choices=["iris", "wine", "breast_cancer", "digits"])
    src.add_argument("--csv")
    p.add_argument("--target", help="target column(s) for --csv; comma-separated -> fewest classes is used")
    p.add_argument("--out", default="results")
    p.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    p.add_argument("--lrs", nargs="+", type=float, default=list(DEFAULT_LRS))
    p.add_argument("--wds", nargs="+", type=float, default=list(DEFAULT_WDS))
    p.add_argument("--outer", type=int, default=10)
    p.add_argument("--inner", type=int, default=5)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--es-frac", type=float, default=0.1)
    p.add_argument("--supcon-temperature", type=float, default=0.07)
    p.add_argument("--no-relu-after-n2", action="store_true")
    p.add_argument("--min-class-count", type=int, default=10)
    p.add_argument("--select-metric", default="balanced_accuracy")
    p.add_argument("--device", default="cpu", help="cpu (default) | cuda | mps | auto")
    p.add_argument("--threads", type=int, default=1, help="torch CPU threads for this process")
    p.add_argument("--inner-jobs", type=int, default=1, help="parallel processes for the inner grid search")
    p.add_argument("--folds", type=int, nargs="+", help="only run these outer folds (e.g. split across machines)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-models", action="store_true", help="also save weights + fitted preprocessors per fold (off by default: disk)")
    p.add_argument("--quick", action="store_true", help="smoke test: 3 outer, 2 inner, 2x2 grid, 5 epochs")
    a = p.parse_args(argv)

    if a.quick:
        a.outer, a.inner, a.epochs, a.lrs, a.wds = 3, 2, 5, a.lrs[1:3], a.wds[:2]
    torch.set_num_threads(a.threads)

    if a.task_id is not None:
        ds = load_openml_task(a.task_id, a.min_class_count)
        tag = f"task{a.task_id}_{ds.name}"
    elif a.sklearn:
        ds = load_sklearn(a.sklearn, a.min_class_count)
        tag = f"sklearn_{a.sklearn}"
    else:
        if not a.target:
            p.error("--csv needs --target")
        ds = load_csv(a.csv, a.target, a.min_class_count)
        tag = f"csv_{ds.name}"
    tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", tag)
    out_dir = os.path.join(a.out, tag)

    cfg = TrainConfig(epochs=a.epochs, patience=a.patience, batch_size=a.batch_size, es_frac=a.es_frac,
                      supcon_temperature=a.supcon_temperature, relu_after_n2=not a.no_relu_after_n2,
                      device=resolve_device(a.device))
    i = ds.info
    print(f"[{ds.name}] n={i['n']} (raw {i['n_raw']}, -{i['n_dropped_missing']} missing, "
          f"-{i['n_dropped_rare']} rare-class) d_enc={i['d_encoded']} C={i['n_classes']} "
          f"target={i['target']} device={cfg.device} -> {out_dir}", flush=True)
    t0 = time.time()
    df = run_nested_cv(ds, out_dir, methods=a.methods, lrs=a.lrs, wds=a.wds, cfg=cfg, n_outer=a.outer,
                       n_inner=a.inner, seed=a.seed, select_metric=a.select_metric, inner_jobs=a.inner_jobs,
                       threads=a.threads, save_models=a.save_models, folds=a.folds,
                       log=lambda s: print(s, flush=True))
    if len(df):
        print(df.groupby("method")[["accuracy", "balanced_accuracy", "f1_macro"]].mean().round(4).to_string())
    print(f"[{ds.name}] total {time.time() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
