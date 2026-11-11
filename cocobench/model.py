"""Shared MLP used by all four losses.

d -> N1 -> ReLU -> N2 -> ReLU -> N3 (=C) -> [L2 normalisation]
N1 = min(512, max(4d, 10)), N3 = C, N2 = (N1 + N3) // 2.

The ReLU after N2 is on by default (two stacked linear layers without a
non-linearity collapse into one); pass relu_after_n2=False for the literal
"N1 -> relu -> N2 -> N3" reading.
"""
import torch.nn as nn
import torch.nn.functional as F


def layer_sizes(d, C):
    n1 = min(512, max(4 * d, 10))
    n3 = C
    n2 = (n1 + n3) // 2
    return n1, n2, n3


class MLP(nn.Module):
    def __init__(self, d, C, normalize_output=False, relu_after_n2=True):
        super().__init__()
        n1, n2, n3 = layer_sizes(d, C)
        self.sizes = (d, n1, n2, n3)
        layers = [nn.Linear(d, n1), nn.ReLU(), nn.Linear(n1, n2)]
        if relu_after_n2:
            layers.append(nn.ReLU())
        layers.append(nn.Linear(n2, n3))
        self.net = nn.Sequential(*layers)
        self.normalize_output = normalize_output

    def forward(self, x):
        z = self.net(x)
        if self.normalize_output:
            z = F.normalize(z, dim=1)
        return z
