import itertools

import pytest
import torch
import torch.nn.functional as F

from cocobench.losses import SupConLoss, CoCoLoss, DotRegressionLoss, simplex_etf, CELoss
from cocobench.model import MLP, layer_sizes


@pytest.mark.parametrize("C,d", [(2, 2), (3, 3), (5, 5), (10, 12), (26, 26)])
def test_simplex_etf_geometry(C, d):
    W = simplex_etf(C, d)
    assert W.shape == (C, d)
    G = W @ W.T
    assert torch.allclose(torch.diag(G), torch.ones(C), atol=1e-5)
    off = G[~torch.eye(C, dtype=torch.bool)]
    assert torch.allclose(off, torch.full_like(off, -1.0 / (C - 1)), atol=1e-5)
    assert torch.allclose(W.sum(0), torch.zeros(d), atol=1e-5)   # centred


def test_etf_needs_dim():
    with pytest.raises(ValueError):
        simplex_etf(5, 3)


@pytest.mark.parametrize("C", [2, 3, 7])
def test_coco_zero_at_etf_optimum(C):
    W = simplex_etf(C, C)
    y1 = torch.randint(0, C, (32,)); y2 = torch.randint(0, C, (32,))
    loss = CoCoLoss(C)(W[y1], W[y2], y1, y2)
    assert loss.item() < 1e-10
    loss_n = CoCoLoss(C, normalize=True)(3.0 * W[y1], 0.5 * W[y2], y1, y2)   # scale-invariant when normalised
    assert loss_n.item() < 1e-10


def test_coco_matches_formula_and_has_gradients():
    torch.manual_seed(0)
    C, B, d = 4, 16, 4
    z1 = torch.randn(B, d, requires_grad=True); z2 = torch.randn(B, d, requires_grad=True)
    y1 = torch.randint(0, C, (B,)); y2 = torch.randint(0, C, (B,))
    loss = CoCoLoss(C)(z1, z2, y1, y2)
    # explicit double loop
    tot = 0.0
    for i, j in itertools.product(range(B), range(B)):
        g = 1.0 if y1[i] == y2[j] else -1.0 / (C - 1)
        tot += (z1[i] @ z2[j] - g) ** 2
    ref = tot / B ** 2 + ((z1.pow(2).sum(1) - 1) ** 2).mean() + ((z2.pow(2).sum(1) - 1) ** 2).mean()
    assert torch.allclose(loss, ref, atol=1e-5)
    loss.backward()
    assert z1.grad.abs().sum() > 0 and z2.grad.abs().sum() > 0


def test_coco_binary_target_is_minus_one():
    l = CoCoLoss(2)
    T = l.target_gram(torch.tensor([0, 1]), torch.tensor([0, 1]))
    assert torch.allclose(T, torch.tensor([[1.0, -1.0], [-1.0, 1.0]]))


def _supcon_reference(z, y, t):
    z = F.normalize(z, dim=1)
    B = len(y)
    num, cnt = 0.0, 0
    for i in range(B):
        den = sum(torch.exp(z[i] @ z[a] / t) for a in range(B) if a != i)
        for p in range(B):
            if p != i and y[p] == y[i]:
                num += -torch.log(torch.exp(z[i] @ z[p] / t) / den); cnt += 1
    return num / cnt


def test_supcon_matches_reference():
    torch.manual_seed(1)
    z = torch.randn(12, 5); y = torch.randint(0, 3, (12,))
    assert torch.allclose(SupConLoss(0.1)(z, y), _supcon_reference(z, y, 0.1), atol=1e-4)


def test_supcon_no_positives_is_zero_with_grad():
    z = torch.randn(3, 4, requires_grad=True)
    loss = SupConLoss()(z, torch.tensor([0, 1, 2]))
    assert loss.item() == 0.0
    loss.backward()


def test_dot_regression():
    C = 5
    W = simplex_etf(C, C)
    y = torch.arange(C)
    dr = DotRegressionLoss(W)
    assert dr(2.0 * W[y], y).item() < 1e-10
    assert dr(-W[y], y).item() == pytest.approx(2.0, abs=1e-5)   # 0.5 * (-1 - 1)^2


def test_ce():
    z = torch.tensor([[10.0, -10.0]]); assert CELoss()(z, torch.tensor([0])).item() < 1e-6


@pytest.mark.parametrize("d,C", [(1, 2), (4, 3), (20, 10), (200, 5)])
def test_architecture_sizes(d, C):
    n1, n2, n3 = layer_sizes(d, C)
    assert n1 == min(512, max(4 * d, 10)) and n3 == C and n2 == (n1 + n3) // 2
    m = MLP(d, C)
    lin = [l for l in m.net if isinstance(l, torch.nn.Linear)]
    assert [(l.in_features, l.out_features) for l in lin] == [(d, n1), (n1, n2), (n2, n3)]
    assert sum(isinstance(l, torch.nn.ReLU) for l in m.net) == 2
    assert sum(isinstance(l, torch.nn.ReLU) for l in MLP(d, C, relu_after_n2=False).net) == 1


def test_normalized_output():
    m = MLP(6, 4, normalize_output=True)
    z = m(torch.randn(10, 6))
    assert torch.allclose(z.norm(dim=1), torch.ones(10), atol=1e-5)


@pytest.mark.parametrize("C", [2, 3, 6])
def test_coco_predictor_is_class_mean_rule(C):
    """Proposition 'h is the classifier': at the CoCo optimum h(x_i)=s_{y_i} (simplex),
    argmax <h, s_k> = y_i with margin C/(C-1)."""
    from cocobench.train import Predictor, TrainConfig
    W = simplex_etf(C, C)
    y = torch.arange(C).repeat(5)
    X = W[y]                                   # identity "network": embeddings at the optimum
    ident = torch.nn.Identity()
    p = Predictor("coco", C, TrainConfig(), "cpu").fit(ident, X, y)
    assert torch.allclose(p.means, W, atol=1e-6)
    s = p.scores(ident, X)
    assert torch.equal(s.argmax(1), y)
    correct = s.gather(1, y[:, None]).squeeze(1)
    other = s.masked_fill(torch.nn.functional.one_hot(y, C).bool(), -1e9).max(1).values
    assert torch.allclose(correct - other, torch.full((len(y),), C / (C - 1)), atol=1e-5)
