"""Training (with early stopping), prediction rules and metrics."""
import copy
import time
from dataclasses import dataclass, asdict

import numpy as np
import torch
from sklearn import metrics as skm
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit

from .losses import CELoss, SupConLoss, CoCoLoss, DotRegressionLoss, simplex_etf
from .model import MLP

METHODS = ("ce", "supcon", "coco", "dr")


@dataclass
class TrainConfig:
    epochs: int = 100
    patience: int = 5
    batch_size: int = 32
    es_frac: float = 0.1            # fraction of the training split held out for early stopping
    es_metric: str = "balanced_accuracy"
    supcon_temperature: float = 0.07
    pred_temperature: float = 0.1   # softmax temperature turning similarity scores into probabilities (AUC only)
    relu_after_n2: bool = True
    device: str = "cpu"


def resolve_device(name):
    if name == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return name


def es_split(y, frac, seed):
    """Stratified train/early-stopping split of indices 0..n-1."""
    n = len(y)
    n_val = max(int(round(frac * n)), 1)
    try:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=max(n_val, len(np.unique(y))), random_state=seed)
        tr, va = next(sss.split(np.zeros(n), y))
    except ValueError:
        tr, va = next(ShuffleSplit(n_splits=1, test_size=n_val, random_state=seed).split(np.zeros(n)))
    return tr, va


def uses_normalized_output(method, coco_norm):
    return method in ("supcon", "dr") or (method == "coco" and bool(coco_norm))


def make_criterion(method, C, coco_norm, cfg):
    if method == "ce":
        return CELoss()
    if method == "supcon":
        return SupConLoss(cfg.supcon_temperature)
    if method == "coco":
        return CoCoLoss(C, normalize=bool(coco_norm))
    if method == "dr":
        return DotRegressionLoss(simplex_etf(C, C))
    raise ValueError(method)


class Predictor:
    """Turns network outputs into class scores.

    ce     : scores = logits
    dr     : scores = <h, w_c> with the fixed simplex-ETF prototypes W (C x C)
    supcon : scores = cosine to the class means of the (normalised) training embeddings
    coco   : scores = <h(x), s_k>, s_k = class mean of the training embeddings h
             (Proposition "h is the classifier": argmax_k <h, s_k> = argmin_k ||h - s_k|| = y)
    """

    def __init__(self, method, C, cfg, device):
        self.method, self.C, self.cfg, self.device = method, C, cfg, device
        self.W = simplex_etf(C, C).to(device) if method == "dr" else None
        self.means = None

    @torch.no_grad()
    def fit(self, model, X, y):
        if self.method not in ("supcon", "coco"):
            return self
        z = embed(model, X)
        if self.method == "supcon":
            z = torch.nn.functional.normalize(z, dim=1)
        sums = torch.zeros(self.C, z.size(1), device=z.device)
        sums.index_add_(0, y, z)
        counts = torch.bincount(y, minlength=self.C).clamp_min(1).to(z.dtype)
        means = sums / counts[:, None]
        self.means = torch.nn.functional.normalize(means, dim=1) if self.method == "supcon" else means
        return self

    @torch.no_grad()
    def scores(self, model, X):
        z = embed(model, X)
        if self.method == "ce":
            return z
        if self.method == "dr":
            return z @ self.W.T
        if self.method == "coco":
            return z @ self.means.T
        return torch.nn.functional.normalize(z, dim=1) @ self.means.T

    @torch.no_grad()
    def predict(self, model, X):
        s = self.scores(model, X)
        if self.method == "ce":
            p = torch.softmax(s, dim=1)
        else:
            p = torch.softmax(s / self.cfg.pred_temperature, dim=1)
        return s.argmax(1).cpu().numpy(), p.cpu().numpy(), s.cpu().numpy()

    def state(self):
        return {"method": self.method, "C": self.C,
                "W": None if self.W is None else self.W.cpu(),
                "means": None if self.means is None else self.means.cpu()}


@torch.no_grad()
def embed(model, X, chunk=8192):
    model.eval()
    return torch.cat([model(X[i:i + chunk]) for i in range(0, len(X), chunk)], 0)


def compute_metrics(y, pred, prob, C):
    labels = list(range(C))
    out = {
        "accuracy": skm.accuracy_score(y, pred),
        "balanced_accuracy": skm.balanced_accuracy_score(y, pred),
        "f1_macro": skm.f1_score(y, pred, average="macro", labels=labels, zero_division=0),
        "f1_weighted": skm.f1_score(y, pred, average="weighted", labels=labels, zero_division=0),
        "precision_macro": skm.precision_score(y, pred, average="macro", labels=labels, zero_division=0),
        "precision_weighted": skm.precision_score(y, pred, average="weighted", labels=labels, zero_division=0),
        "recall_macro": skm.recall_score(y, pred, average="macro", labels=labels, zero_division=0),
        "recall_weighted": skm.recall_score(y, pred, average="weighted", labels=labels, zero_division=0),
        "mcc": skm.matthews_corrcoef(y, pred),
        "cohen_kappa": skm.cohen_kappa_score(y, pred, labels=labels),
    }
    try:
        if C == 2:
            out["roc_auc"] = skm.roc_auc_score(y, prob[:, 1])
        else:
            p = prob / prob.sum(1, keepdims=True)
            out["roc_auc"] = skm.roc_auc_score(y, p, multi_class="ovr", average="macro", labels=labels)
    except ValueError:
        out["roc_auc"] = float("nan")
    return {k: float(v) for k, v in out.items()}


def _batches(n, bs, gen):
    """Equivalent of DataLoader(shuffle=True, drop_last=True) over indices."""
    perm = torch.randperm(n, generator=gen)
    for b in range(n // bs):
        yield perm[b * bs:(b + 1) * bs]


def train_model(method, X_tr, y_tr, X_es, y_es, C, lr, wd, coco_norm, cfg: TrainConfig, seed):
    """Train one network with early stopping on (X_es, y_es). Inputs are numpy arrays.

    Returns (model, predictor, info). The model holds the best (early-stopped) weights.
    """
    device = cfg.device
    torch.manual_seed(seed)
    gen1 = torch.Generator().manual_seed(seed)
    gen2 = torch.Generator().manual_seed(seed + 7919)   # independent second loader for CoCo

    Xt = torch.as_tensor(X_tr, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y_tr, dtype=torch.long, device=device)
    Xe = torch.as_tensor(X_es, dtype=torch.float32, device=device)
    ye_np = np.asarray(y_es)

    d = Xt.shape[1]
    model = MLP(d, C, normalize_output=uses_normalized_output(method, coco_norm),
                relu_after_n2=cfg.relu_after_n2).to(device)
    crit = make_criterion(method, C, coco_norm, cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    pred = Predictor(method, C, cfg, device)

    n = len(Xt)
    bs = min(cfg.batch_size, n)
    best, best_state, best_pred, best_epoch, bad = -np.inf, None, None, 0, 0
    history = []
    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        tot, nb = 0.0, 0
        if method == "coco":
            it = zip(_batches(n, bs, gen1), _batches(n, bs, gen2))
            for i1, i2 in it:
                loss = crit(model(Xt[i1]), model(Xt[i2]), yt[i1], yt[i2])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                tot += loss.item(); nb += 1
        else:
            for i1 in _batches(n, bs, gen1):
                loss = crit(model(Xt[i1]), yt[i1])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                tot += loss.item(); nb += 1
        train_loss = tot / max(nb, 1)
        if not np.isfinite(train_loss):
            history.append({"epoch": epoch, "train_loss": train_loss, "es_score": float("nan")})
            break

        pred.fit(model, Xt, yt)
        yp, prob, _ = pred.predict(model, Xe)
        score = compute_metrics(ye_np, yp, prob, C)[cfg.es_metric]
        history.append({"epoch": epoch, "train_loss": train_loss, "es_score": score})
        if score > best + 1e-12:
            best, best_epoch, bad = score, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
            best_pred = copy.deepcopy(pred)
        else:
            bad += 1
            if bad >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        pred = best_pred
    else:  # diverged in the first epoch
        pred.fit(model, Xt, yt)
    info = {"best_epoch": best_epoch, "epochs_run": len(history), "es_best": float(best),
            "train_time_s": time.time() - t0, "history": history, "batch_size": bs}
    return model, pred, info
