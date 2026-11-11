"""Nested cross-validation for one dataset and all methods.

outer: StratifiedKFold(n_outer)  -> test metrics (one row per method x fold)
inner: StratifiedKFold(n_inner) on the outer-training part -> grid search
       (lr x weight_decay, plus normalisation on/off for CoCo), selection by mean
       inner balanced accuracy.
Every training run holds out `es_frac` of its training data (stratified) for early
stopping (patience 5, max 100 epochs, best weights restored).

Everything is written under out_dir; finished (method, fold) pairs are skipped on
re-run, so an interrupted dataset resumes where it stopped.
"""
import itertools
import json
import os
import time
from dataclasses import asdict

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

from .data import make_preprocessor
from .train import TrainConfig, train_model, compute_metrics, es_split, METHODS

DEFAULT_LRS = (1e-4, 3e-4, 1e-3, 3e-3)
DEFAULT_WDS = (0.0, 1e-4, 1e-3, 1e-2)
METRIC_COLS = ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "precision_macro",
               "precision_weighted", "recall_macro", "recall_weighted", "mcc", "cohen_kappa", "roc_auc"]


def method_grid(method, lrs, wds):
    norms = (False, True) if method == "coco" else (None,)
    return [{"lr": float(lr), "weight_decay": float(wd), "coco_norm": nm}
            for lr, wd, nm in itertools.product(lrs, wds, norms)]


def _seed(base, *parts):
    s = int(base)
    for p in parts:
        s = (s * 1000003 + int(p) + 1) % (2 ** 31 - 1)
    return s


def _prepare_split(ds, tr_idx, te_idx, es_frac, seed):
    """Fit preprocessing on tr_idx, transform both parts, carve early-stopping split from training part."""
    pre = make_preprocessor(ds)
    Xtr = pre.fit_transform(ds.X.iloc[tr_idx]).astype(np.float32)
    Xte = pre.transform(ds.X.iloc[te_idx]).astype(np.float32)
    ytr, yte = ds.y[tr_idx], ds.y[te_idx]
    fit_i, es_i = es_split(ytr, es_frac, seed)
    return {"X_fit": Xtr[fit_i], "y_fit": ytr[fit_i], "X_es": Xtr[es_i], "y_es": ytr[es_i],
            "X_test": Xte, "y_test": yte, "pre": pre}


def _inner_task(method, hp, split, C, cfg, seed, threads):
    torch.set_num_threads(threads)
    model, pred, info = train_model(method, split["X_fit"], split["y_fit"], split["X_es"], split["y_es"],
                                    C, hp["lr"], hp["weight_decay"], hp["coco_norm"], cfg, seed)
    yp, prob, _ = pred.predict(model, torch.as_tensor(split["X_test"], device=cfg.device))
    m = compute_metrics(split["y_test"], yp, prob, C)
    return {**m, "best_epoch": info["best_epoch"], "epochs_run": info["epochs_run"],
            "train_time_s": info["train_time_s"]}


def run_nested_cv(ds, out_dir, methods=METHODS, lrs=DEFAULT_LRS, wds=DEFAULT_WDS,
                  cfg: TrainConfig = None, n_outer=10, n_inner=5, seed=0, select_metric="balanced_accuracy",
                  inner_jobs=1, threads=1, save_models=False, folds=None, log=print):
    cfg = cfg or TrainConfig()
    C = ds.n_classes
    os.makedirs(out_dir, exist_ok=True)
    subs = ["fold_results", "predictions", "histories"] + (["models", "preprocessors"] if save_models else [])
    for sub in subs:
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    run_cfg = {"methods": list(methods), "lrs": list(lrs), "weight_decays": list(wds),
               "n_outer": n_outer, "n_inner": n_inner, "seed": seed, "select_metric": select_metric,
               "train": asdict(cfg), "torch": torch.__version__}
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(run_cfg, f, indent=2)
    with open(os.path.join(out_dir, "dataset_info.json"), "w") as f:
        json.dump(ds.info, f, indent=2, default=str)

    outer = list(StratifiedKFold(n_outer, shuffle=True, random_state=seed).split(np.zeros(len(ds.y)), ds.y))
    fold_of = np.empty(len(ds.y), dtype=int)
    for k, (_, te) in enumerate(outer):
        fold_of[te] = k
    pd.DataFrame({"sample": np.arange(len(ds.y)), "y": ds.y, "outer_fold": fold_of}).to_csv(
        os.path.join(out_dir, "outer_folds.csv"), index=False)

    def done_path(method, k):
        return os.path.join(out_dir, "fold_results", f"{method}_fold{k}.json")

    for k, (tr, te) in enumerate(outer):
        if folds is not None and k not in folds:
            continue
        todo = [m for m in methods if not os.path.exists(done_path(m, k))]
        if not todo:
            log(f"[{ds.name}] outer fold {k}: done, skipping")
            continue
        t_fold = time.time()
        # ---- inner splits (preprocessing fitted on each inner-training part)
        inner = list(StratifiedKFold(n_inner, shuffle=True, random_state=_seed(seed, k))
                     .split(np.zeros(len(tr)), ds.y[tr]))
        inner_splits = [_prepare_split(ds, tr[a], tr[b], cfg.es_frac, _seed(seed, k, j))
                        for j, (a, b) in enumerate(inner)]
        outer_split = _prepare_split(ds, tr, te, cfg.es_frac, _seed(seed, k, 99))
        if save_models:
            joblib.dump(outer_split["pre"], os.path.join(out_dir, "preprocessors", f"fold{k}.joblib"))

        for method in todo:
            t_m = time.time()
            grid = method_grid(method, lrs, wds)
            jobs = [(ci, j) for ci in range(len(grid)) for j in range(n_inner)]
            if inner_jobs == 1:
                res = [_inner_task(method, grid[ci], inner_splits[j], C, cfg, _seed(seed, k, j, ci), threads)
                       for ci, j in jobs]
            else:
                res = joblib.Parallel(n_jobs=inner_jobs, prefer="processes")(
                    joblib.delayed(_inner_task)(method, grid[ci], inner_splits[j], C, cfg,
                                                _seed(seed, k, j, ci), 1) for ci, j in jobs)
            inner_rows = [{"method": method, "outer_fold": k, "config_id": ci, "inner_fold": j,
                           "lr": grid[ci]["lr"], "weight_decay": grid[ci]["weight_decay"],
                           "coco_norm": grid[ci]["coco_norm"], **r} for (ci, j), r in zip(jobs, res)]
            idf = pd.DataFrame(inner_rows)
            mean_sel = idf.groupby("config_id")[select_metric].mean()
            best_ci = int(mean_sel.idxmax())        # ties -> first config (smallest lr, wd)
            hp = grid[best_ci]

            # ---- refit on the whole outer-training part, test on outer test fold
            model, pred, info = train_model(method, outer_split["X_fit"], outer_split["y_fit"],
                                            outer_split["X_es"], outer_split["y_es"], C, hp["lr"],
                                            hp["weight_decay"], hp["coco_norm"], cfg, _seed(seed, k, 999))
            Xte = torch.as_tensor(outer_split["X_test"], device=cfg.device)
            yp, prob, scores = pred.predict(model, Xte)
            m = compute_metrics(outer_split["y_test"], yp, prob, C)

            pdf = pd.DataFrame({"method": method, "outer_fold": k, "sample": te,
                                "y_true": outer_split["y_test"], "y_pred": yp})
            for c in range(C):
                pdf[f"prob_{c}"] = prob[:, c]
            for c in range(C):
                pdf[f"score_{c}"] = scores[:, c]
            pdf.to_csv(os.path.join(out_dir, "predictions", f"{method}_fold{k}.csv"), index=False)
            pd.DataFrame(info["history"]).to_csv(
                os.path.join(out_dir, "histories", f"{method}_fold{k}.csv"), index=False)
            if save_models:
                torch.save({"state_dict": {a: b.cpu() for a, b in model.state_dict().items()},
                            "sizes": model.sizes, "normalize_output": model.normalize_output,
                            "relu_after_n2": cfg.relu_after_n2, "hyperparams": hp,
                            "predictor": pred.state(), "classes": ds.classes,
                            "preprocessor": f"preprocessors/fold{k}.joblib"},
                           os.path.join(out_dir, "models", f"{method}_fold{k}.pt"))
            result = {"dataset": ds.name, "method": method, "outer_fold": k,
                      "n_train": int(len(tr)), "n_test": int(len(te)), "n_classes": C,
                      "best_config_id": best_ci, **{f"best_{a}": b for a, b in hp.items()},
                      "inner_best_" + select_metric: float(mean_sel.max()),
                      "best_epoch": info["best_epoch"], "epochs_run": info["epochs_run"],
                      "final_train_time_s": info["train_time_s"], "method_time_s": time.time() - t_m,
                      "test": m, "inner": inner_rows}
            with open(done_path(method, k) + ".tmp", "w") as f:
                json.dump(result, f, default=float)
            os.replace(done_path(method, k) + ".tmp", done_path(method, k))
            log(f"[{ds.name}] fold {k} {method:6s} best={hp} test BA={m['balanced_accuracy']:.4f} "
                f"({time.time() - t_m:.1f}s)")
        log(f"[{ds.name}] outer fold {k} finished in {time.time() - t_fold:.1f}s")

    return assemble(out_dir, ds.name, methods, n_outer)


def assemble(out_dir, name, methods, n_outer):
    """Collect per-fold JSONs into the CSV files."""
    outer_rows, best_rows, inner_rows = [], [], []
    for method in methods:
        for k in range(n_outer):
            p = os.path.join(out_dir, "fold_results", f"{method}_fold{k}.json")
            if not os.path.exists(p):
                continue
            with open(p) as f:
                r = json.load(f)
            base = {"dataset": name, "method": method, "outer_fold": k}
            outer_rows.append({**base, **r["test"], "best_epoch": r["best_epoch"],
                               "epochs_run": r["epochs_run"], "n_train": r["n_train"], "n_test": r["n_test"],
                               "final_train_time_s": r["final_train_time_s"]})
            best_rows.append({**base, **{a: b for a, b in r.items() if a.startswith("best_") or
                                         a.startswith("inner_best_")}, "method_time_s": r["method_time_s"]})
            inner_rows.extend({"dataset": name, **row} for row in r["inner"])
    outer_df = pd.DataFrame(outer_rows)
    outer_df.to_csv(os.path.join(out_dir, "outer_metrics.csv"), index=False)
    pd.DataFrame(best_rows).to_csv(os.path.join(out_dir, "best_params.csv"), index=False)
    pd.DataFrame(inner_rows).to_csv(os.path.join(out_dir, "inner_cv.csv"), index=False)
    preds = [os.path.join(out_dir, "predictions", f"{m}_fold{k}.csv") for m in methods for k in range(n_outer)]
    preds = [p for p in preds if os.path.exists(p)]
    if preds:
        pd.concat([pd.read_csv(p) for p in preds]).to_csv(os.path.join(out_dir, "predictions.csv"), index=False)
    if len(outer_df):
        cols = [c for c in METRIC_COLS if c in outer_df]
        summ = outer_df.groupby("method")[cols].agg(["mean", "std"])
        summ.columns = [f"{a}_{b}" for a, b in summ.columns]
        summ.insert(0, "n_folds", outer_df.groupby("method").size())
        summ.to_csv(os.path.join(out_dir, "summary.csv"))
    complete = len(outer_df) == len(methods) * n_outer
    if complete:
        open(os.path.join(out_dir, "DONE"), "w").close()
    return outer_df
