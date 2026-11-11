"""Dataset loading and preprocessing.

Steps (applied to the full table, no label/statistic leakage):
  1. choose ONE target (if several candidate targets, the one with fewest classes);
     the other target columns are dropped from the features;
  2. drop every sample (row) that has a missing value;
  3. drop constant feature columns;
  4. drop classes with fewer than `min_class_count` samples (needed for stratified 10-fold CV);
  5. encode labels to 0..C-1.

Per-fold (fitted on the training part only, see `make_preprocessor`):
  numeric -> StandardScaler, categorical -> one-hot (vocabulary from full table).
"""
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class Dataset:
    name: str
    X: pd.DataFrame
    y: np.ndarray                # int labels 0..C-1
    classes: list                # original class names
    num_cols: list
    cat_cols: list
    categories: dict             # col -> sorted list of categories (vocabulary)
    info: dict = field(default_factory=dict)

    @property
    def n_classes(self):
        return len(self.classes)


def choose_target(df, candidates):
    """Pick the candidate target with the fewest distinct (non-missing) values."""
    candidates = [c for c in candidates if c in df.columns]
    if not candidates:
        raise ValueError("no target column found")
    nunique = {c: df[c].dropna().nunique() for c in candidates}
    return min(candidates, key=lambda c: (nunique[c], candidates.index(c))), nunique


def preprocess_frame(df, target_candidates, name="dataset", categorical=None, min_class_count=10):
    """df: full table (features + target(s)). categorical: optional list of categorical feature names."""
    if isinstance(target_candidates, str):
        target_candidates = [t.strip() for t in target_candidates.split(",") if t.strip()]
    info = {"name": name, "n_raw": int(len(df)), "target_candidates": list(target_candidates)}
    target, nunique = choose_target(df, target_candidates)
    info["target"] = target
    info["target_nunique"] = {k: int(v) for k, v in nunique.items()}

    feats = df.drop(columns=[c for c in target_candidates if c in df.columns])
    y_raw = df[target]

    # 2. remove samples with missing values (features or target)
    keep = feats.notna().all(axis=1) & y_raw.notna()
    feats, y_raw = feats.loc[keep], y_raw.loc[keep]
    info["n_dropped_missing"] = int((~keep).sum())

    # 3. constant columns
    const = [c for c in feats.columns if feats[c].nunique(dropna=False) <= 1]
    feats = feats.drop(columns=const)
    info["dropped_constant_cols"] = const

    # 4. rare classes
    counts = y_raw.value_counts()
    rare = counts[counts < min_class_count].index.tolist()
    if rare:
        m = ~y_raw.isin(rare)
        feats, y_raw = feats.loc[m], y_raw.loc[m]
    info["dropped_rare_classes"] = [str(r) for r in rare]
    info["n_dropped_rare"] = int(counts[rare].sum()) if rare else 0

    # categorical vs numeric
    if categorical is None:
        categorical = []
    cat_cols = [c for c in feats.columns
                if c in set(categorical)
                or isinstance(feats[c].dtype, pd.CategoricalDtype)
                or feats[c].dtype == object or feats[c].dtype == bool
                or pd.api.types.is_string_dtype(feats[c])]
    num_cols = [c for c in feats.columns if c not in cat_cols]
    feats = feats.copy()
    for c in cat_cols:
        feats[c] = feats[c].astype(str)
    for c in num_cols:
        feats[c] = pd.to_numeric(feats[c], errors="raise").astype(np.float64)
    categories = {c: sorted(feats[c].unique().tolist()) for c in cat_cols}

    classes = sorted(pd.unique(y_raw.astype(str)).tolist())
    cmap = {c: i for i, c in enumerate(classes)}
    y = y_raw.astype(str).map(cmap).to_numpy(dtype=np.int64)

    feats = feats.reset_index(drop=True)
    info.update({
        "n": int(len(feats)), "n_num": len(num_cols), "n_cat": len(cat_cols),
        "n_classes": len(classes),
        "class_counts": {cl: int((y == i).sum()) for i, cl in enumerate(classes)},
        "d_encoded": len(num_cols) + sum(len(v) for v in categories.values()),
    })
    if len(classes) < 2:
        raise ValueError(f"{name}: fewer than 2 classes after preprocessing")
    return Dataset(name, feats, y, classes, num_cols, cat_cols, categories, info)


def make_preprocessor(ds: Dataset):
    """Unfitted transformer; fit it on the training fold only."""
    transformers = []
    if ds.num_cols:
        transformers.append(("num", StandardScaler(), ds.num_cols))
    if ds.cat_cols:
        cats = [ds.categories[c] for c in ds.cat_cols]
        try:
            ohe = OneHotEncoder(categories=cats, handle_unknown="ignore", sparse_output=False)
        except TypeError:  # sklearn < 1.2
            ohe = OneHotEncoder(categories=cats, handle_unknown="ignore", sparse=False)
        transformers.append(("cat", ohe, ds.cat_cols))
    return ColumnTransformer(transformers, remainder="drop")


# ---------------------------------------------------------------- loaders
def load_openml_task(task_id, min_class_count=10):
    import openml
    try:
        task = openml.tasks.get_task(task_id, download_splits=False, download_data=True,
                                     download_qualities=False)
    except TypeError:
        task = openml.tasks.get_task(task_id)
    dataset = task.get_dataset()
    try:
        df, _, cat_ind, names = dataset.get_data(dataset_format="dataframe", target=None)
    except TypeError:
        df, _, cat_ind, names = dataset.get_data(target=None)
    targets = getattr(task, "target_name", None) or dataset.default_target_attribute
    if isinstance(targets, (list, tuple)):
        targets = ",".join(targets)
    # also consider the dataset's default targets (may list several)
    cands = [t.strip() for t in str(targets).split(",")]
    if dataset.default_target_attribute:
        for t in str(dataset.default_target_attribute).split(","):
            if t.strip() and t.strip() not in cands:
                cands.append(t.strip())
    ignore = dataset.ignore_attribute or []
    if isinstance(ignore, str):
        ignore = [ignore]
    row_id = [dataset.row_id_attribute] if dataset.row_id_attribute else []
    df = df.drop(columns=[c for c in ignore + row_id if c in df.columns and c not in cands])
    categorical = [n for n, c in zip(names, cat_ind) if c and n not in cands]
    ds = preprocess_frame(df, cands, name=dataset.name, categorical=categorical,
                          min_class_count=min_class_count)
    ds.info.update({"openml_task_id": int(task_id), "openml_dataset_id": int(dataset.dataset_id)})
    return ds


def load_sklearn(name, min_class_count=10):
    """Bundled sklearn datasets (no internet) for smoke tests: iris, wine, breast_cancer, digits."""
    from sklearn import datasets
    loader = {"iris": datasets.load_iris, "wine": datasets.load_wine,
              "breast_cancer": datasets.load_breast_cancer, "digits": datasets.load_digits}[name]
    b = loader(as_frame=True)
    df = b.frame.copy()
    df["target"] = df["target"].astype(str)
    return preprocess_frame(df, ["target"], name=name, min_class_count=min_class_count)


def load_csv(path, target, min_class_count=10):
    df = pd.read_csv(path)
    import os
    return preprocess_frame(df, target, name=os.path.splitext(os.path.basename(path))[0],
                            min_class_count=min_class_count)
