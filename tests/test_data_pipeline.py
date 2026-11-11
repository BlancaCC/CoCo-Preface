import json
import os

import numpy as np
import pandas as pd
import pytest
import torch

from cocobench.data import preprocess_frame, make_preprocessor, load_sklearn
from cocobench.nested_cv import run_nested_cv, method_grid
from cocobench.train import TrainConfig, train_model, _batches, es_split, compute_metrics


def _toy_frame(n=200, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, n)
    df = pd.DataFrame({
        "num1": rng.normal(size=n) + y,
        "num2": rng.normal(size=n) * 5 + 100,
        "const": 1.0,
        "cat": pd.Categorical(rng.choice(["a", "b", "c"], n)),
        "label": np.array(["x", "y", "z"])[y],
        "other_target": rng.integers(0, 6, n),
    })
    df.loc[[3, 7, 11], "num1"] = np.nan
    df.loc[[5], "cat"] = np.nan
    df.loc[[9], "label"] = None
    return df


def test_preprocess_target_choice_missing_constant():
    df = _toy_frame()
    ds = preprocess_frame(df, "other_target,label", name="toy")
    assert ds.info["target"] == "label"                      # fewer classes than other_target
    assert "other_target" not in ds.X.columns and "label" not in ds.X.columns
    assert ds.info["n_dropped_missing"] == 5
    assert len(ds.X) == 195 and not ds.X.isna().any().any()
    assert "const" not in ds.X.columns
    assert ds.cat_cols == ["cat"] and set(ds.num_cols) == {"num1", "num2"}
    assert ds.n_classes == 3 and set(np.unique(ds.y)) == {0, 1, 2}
    assert ds.info["d_encoded"] == 2 + 3


def test_rare_classes_dropped():
    df = _toy_frame()
    df.loc[:4, "label"] = "rare"
    ds = preprocess_frame(df, ["label"], min_class_count=10)
    assert "rare" in ds.info["dropped_rare_classes"] and "rare" not in ds.classes


def test_preprocessor_fit_on_train_only():
    ds = preprocess_frame(_toy_frame(), ["label"])
    tr, te = np.arange(0, 100), np.arange(100, len(ds.y))
    pre = make_preprocessor(ds)
    Xtr = pre.fit_transform(ds.X.iloc[tr]); Xte = pre.transform(ds.X.iloc[te])
    assert np.allclose(Xtr[:, :2].mean(0), 0, atol=1e-8) and np.allclose(Xtr[:, :2].std(0), 1, atol=1e-6)
    assert not np.allclose(Xte[:, :2].mean(0), 0, atol=1e-3)   # test not used for fitting
    assert Xtr.shape[1] == Xte.shape[1] == ds.info["d_encoded"]


def test_double_loader_independent_and_complete():
    g1, g2 = torch.Generator().manual_seed(0), torch.Generator().manual_seed(1)
    b1, b2 = list(_batches(100, 32, g1)), list(_batches(100, 32, g2))
    assert len(b1) == len(b2) == 3 and all(len(b) == 32 for b in b1)   # drop_last
    assert len(set(torch.cat(b1).tolist())) == 96                        # no repetition in an epoch
    assert not all(torch.equal(a, b) for a, b in zip(b1, b2))            # two different shuffles


def test_es_split_stratified():
    y = np.repeat([0, 1, 2], [50, 30, 20])
    tr, va = es_split(y, 0.1, 0)
    assert len(set(tr) & set(va)) == 0 and len(tr) + len(va) == 100
    assert set(y[va]) == {0, 1, 2}


def test_grid_sizes():
    assert len(method_grid("ce", [1, 2, 3, 4], [1, 2, 3, 4])) == 16
    assert len(method_grid("coco", [1, 2, 3, 4], [1, 2, 3, 4])) == 32


def test_metrics_perfect():
    y = np.array([0, 1, 2, 0]); p = np.eye(3)[y]
    m = compute_metrics(y, y, p, 3)
    assert all(m[k] == pytest.approx(1.0) for k in ["accuracy", "balanced_accuracy", "f1_macro", "mcc", "roc_auc"])


@pytest.mark.parametrize("method", ["ce", "supcon", "coco", "dr"])
def test_each_method_learns_iris(method):
    ds = load_sklearn("iris")
    pre = make_preprocessor(ds); X = pre.fit_transform(ds.X).astype(np.float32)
    rng = np.random.default_rng(0); idx = rng.permutation(len(X)); tr, te = idx[:120], idx[120:]
    # Un-normalised CoCo can sit on a plateau for the first epochs on tiny data,
    # so it gets a longer budget in this smoke test (the training code is unchanged).
    cfg = TrainConfig(epochs=200, patience=200) if method == "coco" else TrainConfig(epochs=60, patience=10)
    model, pred, info = train_model(method, X[tr], ds.y[tr], X[te], ds.y[te], 3, 3e-3, 1e-4,
                                    False, cfg, seed=0)
    yp, prob, _ = pred.predict(model, torch.as_tensor(X[te]))
    assert (yp == ds.y[te]).mean() > 0.8, (method, info["best_epoch"])
    assert prob.shape == (30, 3) and np.allclose(prob.sum(1), 1, atol=1e-5)


def test_nested_cv_end_to_end_and_resume(tmp_path):
    ds = load_sklearn("wine")
    out = tmp_path / "wine"
    cfg = TrainConfig(epochs=4, patience=2)
    df = run_nested_cv(ds, str(out), lrs=[1e-3], wds=[0.0, 1e-3], cfg=cfg, n_outer=3, n_inner=2, seed=0,
                       log=lambda s: None, save_models=True)
    assert len(df) == 4 * 3 and (out / "DONE").exists()
    # default: no model files written
    out2 = tmp_path / "nomodels"
    run_nested_cv(ds, str(out2), methods=["ce"], lrs=[1e-3], wds=[0.0], cfg=cfg, n_outer=3, n_inner=2,
                  log=lambda s: None)
    assert not (out2 / "models").exists() and not (out2 / "preprocessors").exists()
    for f in ["outer_metrics.csv", "best_params.csv", "inner_cv.csv", "predictions.csv", "summary.csv",
              "outer_folds.csv", "dataset_info.json", "config.json"]:
        assert (out / f).exists(), f
    folds = pd.read_csv(out / "outer_folds.csv")
    assert folds.outer_fold.nunique() == 3
    # stratified: class proportions per fold close to global
    tab = pd.crosstab(folds.outer_fold, folds.y, normalize="index")
    assert (tab.max() - tab.min()).max() < 0.1
    # every sample predicted exactly once per method
    preds = pd.read_csv(out / "predictions.csv")
    assert (preds.groupby("method")["sample"].nunique() == len(ds.y)).all()
    inner = pd.read_csv(out / "inner_cv.csv")
    assert len(inner[inner.method == "coco"]) == 3 * 2 * 2 * 2      # folds x configs(2wd x 2norm) x inner
    assert len(inner[inner.method == "ce"]) == 3 * 2 * 2
    assert len(list((out / "models").glob("*.pt"))) == 12
    ck = torch.load(out / "models" / "coco_fold0.pt", weights_only=False)
    assert "state_dict" in ck and "hyperparams" in ck
    # resume: nothing re-trained
    mtime = os.path.getmtime(out / "fold_results" / "ce_fold0.json")
    run_nested_cv(ds, str(out), lrs=[1e-3], wds=[0.0, 1e-3], cfg=cfg, n_outer=3, n_inner=2, seed=0,
                  log=lambda s: None)
    assert os.path.getmtime(out / "fold_results" / "ce_fold0.json") == mtime
