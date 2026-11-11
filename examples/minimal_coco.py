#!/usr/bin/env python
"""Minimal, self-contained example of training with the CoCo loss.

Two independent shuffled DataLoaders over the same training set provide the
two batches (x1, y1) and (x2, y2) at every step. The network output itself is
the embedding; prediction assigns each sample to the closest training class mean.

    python examples/minimal_coco.py
"""
import os
import sys

import torch
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cocobench.losses import CoCoLoss  # noqa: E402
from cocobench.model import MLP  # noqa: E402

torch.manual_seed(0)
X, y = load_wine(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
scaler = StandardScaler().fit(X_tr)
X_tr = torch.tensor(scaler.transform(X_tr), dtype=torch.float32)
X_te = torch.tensor(scaler.transform(X_te), dtype=torch.float32)
y_tr, y_te = torch.tensor(y_tr), torch.tensor(y_te)
C = int(y_tr.max()) + 1

train_ds = TensorDataset(X_tr, y_tr)
model = MLP(d=X_tr.shape[1], C=C)                 # output dimension = C, no normalisation
criterion = CoCoLoss(num_classes=C)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for epoch in range(200):
    loader1 = DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=True)
    loader2 = DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=True)
    model.train()
    for (x1, y1), (x2, y2) in zip(loader1, loader2):
        loss = criterion(model(x1), model(x2), y1, y2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# prediction: argmax_k <h(x), s_k>, with s_k the class means of the training embeddings
model.eval()
with torch.no_grad():
    h_tr = model(X_tr)
    means = torch.stack([h_tr[y_tr == k].mean(0) for k in range(C)])
    pred = (model(X_te) @ means.T).argmax(1)
    gram = means @ means.T
print(f"final train loss: {loss.item():.4f}")
print(f"test accuracy:    {(pred == y_te).float().mean().item():.4f}")
print("Gram matrix of the class means (target: 1 on the diagonal, -1/(C-1) elsewhere):")
print(gram.numpy().round(3))
