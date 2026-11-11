import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "analysis"))
import analyze_results as ar  # noqa: E402


def test_holm_known_values():
    adj = ar.holm([0.01, 0.04, 0.03, 0.005])
    assert np.allclose(adj, [0.03, 0.06, 0.06, 0.02])


def test_safe_wilcoxon_identical_is_one():
    assert ar.safe_wilcoxon(np.ones(10), np.ones(10)) == 1.0


def _fake_results(root, n_ds=12, seed=0):
    rng = np.random.default_rng(seed)
    for d in range(n_ds):
        folder = root / f"task{d + 1}_ds{d}"
        (folder / "histories").mkdir(parents=True)
        base = rng.uniform(0.6, 0.9)
        rows = []
        for k in range(10):
            for m, off in [("coco", 0.05), ("ce", 0.0), ("supcon", 0.0), ("dr", -0.05)]:
                v = base + off + rng.normal(0, 0.005)
                rows.append({"dataset": f"ds{d}", "method": m, "outer_fold": k,
                             **{met: v for met in ar.METRICS},
                             "epochs_run": 10, "best_epoch": 5, "final_train_time_s": 1.0})
                pd.DataFrame({"epoch": [1, 2], "train_loss": [1.0, 0.5], "es_score": [0.5, 0.6]}).to_csv(
                    folder / "histories" / f"{m}_fold{k}.csv", index=False)
        pd.DataFrame(rows).to_csv(folder / "outer_metrics.csv", index=False)
        json.dump({"name": f"ds{d}", "n": 100, "d_encoded": 5, "n_classes": 2,
                   "class_counts": {"a": 60 + d, "b": 40}}, open(folder / "dataset_info.json", "w"))
        json.dump({"train": {"supcon_temperature": 0.07}}, open(folder / "config.json", "w"))
        (folder / "DONE").touch()


def test_analysis_end_to_end(tmp_path):
    _fake_results(tmp_path / "results")
    out = tmp_path / "out"
    ar.main(["--results", str(tmp_path / "results"), "--out", str(out)])
    for f in ["report.md", "results_table.tex", "per_dataset_means.csv", "per_dataset_wtl.csv",
              "across_dataset_tests.csv", "imbalance.csv", "training_dynamics.csv"]:
        assert (out / f).exists(), f
    acr = pd.read_csv(out / "across_dataset_tests.csv")
    ba = acr[acr.metric == "balanced_accuracy"]
    assert (ba.coco_better == 12).all() and (ba.wilcoxon_p_holm < 0.05).all()
