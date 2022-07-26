# python peripherals
import numpy
import copy

# torch
import torch
import torchvision
from torchvision import transforms

# lightly
from lightly.data import LightlyDataset
from lightly.data import SimCLRCollateFunction
from lightly.loss import NegativeCosineSimilarity
from lightly.models.modules import BYOLProjectionHead, BYOLPredictionHead
from lightly.models.utils import deactivate_requires_grad
from lightly.models.utils import update_momentum


class BYOL(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()

        self.backbone = backbone
        self.projection_head = BYOLProjectionHead(2048, 4096, 1024)
        self.prediction_head = BYOLPredictionHead(1024, 2048, 1024)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

    def forward(self, x):
        y = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(y)
        p = self.prediction_head(z)
        return p

    def forward_momentum(self, x):
        y = self.backbone_momentum(x).flatten(start_dim=1)
        z = self.projection_head_momentum(y)
        z = z.detach()
        return z


class WSIBYOL(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._resnet = torchvision.models.resnet50()
        self._backbone = torch.nn.Sequential(*list(self._resnet.children())[:-1])

        print('============== RESNET50 ==============')
        print('======================================')
        print(self._resnet)

        print('============== BACKBONE ==============')
        print('======================================')
        print(self._backbone)

        self._model = BYOL(self._backbone)

    def forward(self, in_features):
        update_momentum(self._model.backbone, self._model.backbone_momentum, m=0.99)
        update_momentum(self._model.projection_head, self._model.projection_head_momentum, m=0.99)
        print(f'================== in_features.shape: {in_features.shape} ==================')
        x0 = in_features[:, 0, :, :, :]
        x1 = in_features[:, 1, :, :, :]
        p0 = self._model(x0)
        z0 = self._model.forward_momentum(x0)
        p1 = self._model(x1)
        z1 = self._model.forward_momentum(x1)
        return torch.stack((p0, z0, p1, z1))
