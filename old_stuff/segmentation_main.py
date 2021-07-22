# This file is the main run file for segmentation net

import os
import torch
import pytorch_lightning as pl
from torchvision.models import resnet18
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms


class SegmentationModel(pl.LightningModule):
    def __init__(self):
        super.__init__()
        self.resnet18 = resnet18()

    # forward is not implemented since it is the same as the original of resnet18


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5, weight_decay=5e-5)
        return optimizer

    def training_step(self, batch, batch_idx):
        data, target = batch
        score = self(data)
        loss = F.cross_entropy(score, target)
        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        score = self(data)
        val_loss = F.cross_entropy(score, target)
        return val_loss

    def test_step(self, batch, batch_idx):
        data, target = batch
        score = self(data)
        test_loss = F.cross_entropy(score, target)
        return test_loss



