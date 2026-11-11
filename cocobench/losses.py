"""Loss functions compared in the paper.

All losses receive RAW (un-normalised) network outputs z of shape [B, d].
If the network already L2-normalises its output (SupCon, Dot-Regression, and CoCo
with ``normalize=True``), re-normalising inside the loss is a no-op.

    CELoss              cross-entropy on the C-dimensional logits (baseline)
    SupConLoss          Supervised Contrastive loss, Khosla et al., NeurIPS 2020
    CoCoLoss            the proposed CoCo loss (cross-batch Gram-matrix regression)
    DotRegressionLoss   Dot-Regression with a fixed simplex-ETF classifier,
                        Yang et al., NeurIPS 2022
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CELoss(nn.Module):
    """Plain cross-entropy on the C-dimensional output (logits)."""

    def forward(self, z, y):
        return F.cross_entropy(z, y)


class SupConLoss(nn.Module):
    """Supervised Contrastive loss (Khosla et al., 2020), single-view version.

    Every other sample of the same class in the batch is a positive; all other
    samples (except the anchor itself) appear in the denominator.
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z, y):
        z = F.normalize(z, dim=1)
        logits = z @ z.T / self.temperature
        mask = y[:, None].eq(y[None, :])
        eye = torch.eye(z.size(0), dtype=torch.bool, device=z.device)
        mask = mask & ~eye
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        exp_logits = torch.exp(logits) * (~eye)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
        positives = mask.sum(dim=1)
        valid = positives > 0
        if not valid.any():
            return logits.sum() * 0.0
        return -(log_prob[valid] * mask[valid]).sum() / positives[valid].sum()


class CoCoLoss(nn.Module):
    """CoCo loss on two independently shuffled batches (z, y) and (z', y').

        L = mean_{i,j} ( <z_i, z'_j> - g(y_i, y'_j) )^2
            + mean_i ( ||z_i||^2  - 1 )^2
            + mean_j ( ||z'_j||^2 - 1 )^2

    with g(a, b) = 1 if a == b and -1/(C-1) otherwise, i.e. the target is the Gram
    matrix of a simplex equiangular tight frame (ETF). The embeddings are NOT
    normalised: the two norm terms push them towards the unit sphere.

    ``normalize=True`` L2-normalises z and z' first (the norm terms then vanish);
    it is treated as a hyperparameter in the benchmark.
    """

    def __init__(self, num_classes, normalize=False):
        super().__init__()
        if num_classes < 2:
            raise ValueError("CoCo needs at least 2 classes")
        self.C = num_classes
        self.normalize = normalize
        self.neg = -1.0 / (num_classes - 1)

    def target_gram(self, y1, y2):
        """Target Gram matrix g(y1_i, y2_j), shape [B1, B2]."""
        same = y1[:, None].eq(y2[None, :]).float()
        return same * (1.0 - self.neg) + self.neg

    def forward(self, z1, z2, y1, y2):
        if self.normalize:
            z1 = F.normalize(z1, dim=1)
            z2 = F.normalize(z2, dim=1)
        gram = z1 @ z2.T
        target = self.target_gram(y1, y2).to(gram.dtype)
        align = ((gram - target) ** 2).mean()
        norm1 = ((z1.pow(2).sum(1) - 1.0) ** 2).mean()
        norm2 = ((z2.pow(2).sum(1) - 1.0) ** 2).mean()
        return align + norm1 + norm2


def simplex_etf(C, d):
    """Fixed simplex-ETF prototypes W [C, d]: unit norm, <w_c, w_c'> = -1/(C-1). Needs d >= C-1."""
    if d < C - 1:
        raise ValueError(f"Dot-Regression needs emb_dim >= n_classes-1 ({d} < {C-1}); increase emb_dim.")
    M = torch.eye(C) - 1.0 / C                      # centred one-hot rows
    U, _, _ = torch.linalg.svd(M)
    W = F.normalize(M @ U[:, :C - 1], dim=1)        # [C, C-1]
    return F.pad(W, (0, d - (C - 1)))               # [C, d]


class DotRegressionLoss(nn.Module):
    """Dot-Regression (Yang et al., NeurIPS 2022): 0.5 * (w_y^T h - 1)^2,
    with h = z / ||z|| and a fixed simplex-ETF classifier W."""

    def __init__(self, W):
        super().__init__()
        self.register_buffer("W", W)

    def forward(self, z, y):
        h = F.normalize(z, dim=1)
        return 0.5 * (((h * self.W[y]).sum(1) - 1) ** 2).mean()
