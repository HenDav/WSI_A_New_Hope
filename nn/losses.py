# python peripherals
import itertools

# torch
import torch

# lightly
from lightly.loss import NegativeCosineSimilarity


class TupletLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output_features):
        v = output_features[:, 0, :]
        v2 = v.unsqueeze(dim=1)
        v3 = v2 - output_features
        v4 = v3.norm(dim=2)
        v5 = v4[:, 1:]
        v6 = v5[:, 0]
        v7 = v6.unsqueeze(dim=1)
        v8 = v7 - v5
        v9 = v8[:, 1:]
        v10 = v9.exp()
        v11 = v10.sum(dim=1)
        v12 = v11 + 1
        v13 = v12.log()
        return v13.mean(dim=0)


class BYOLLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._criterion = NegativeCosineSimilarity()

    def forward(self, in_features):
        p0 = in_features[0, :, :]
        z0 = in_features[1, :, :]
        p1 = in_features[2, :, :]
        z1 = in_features[3, :, :]
        return 0.5 * (self._criterion(p0, z1) + self._criterion(p1, z0))
