#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import torch
import torch.nn as nn
from torchvision import models


class MobileNetV3SmallHead(nn.Module):
    """
    MobileNetV3-Small backbone + new classifier head for 29-way classification.
    `transfer=True` uses pretrained weights; set TORCH_HOME to artifacts path before instantiation
    to avoid usage of ~/.cache. You can freeze backbone for N epochs in training.
    """
    def __init__(self, num_classes: int = 29, transfer: bool = True):
        super().__init__()
        # For torchvision>=0.13, use weights=...
        if transfer:
            try:
                weights = models.MobileNet_V3_Small_Weights.DEFAULT
                backbone = models.mobilenet_v3_small(weights=weights)
            except Exception:
                # Fallback older API
                backbone = models.mobilenet_v3_small(pretrained=True)
        else:
            backbone = models.mobilenet_v3_small(weights=None) if hasattr(models, "MobileNet_V3_Small_Weights") else models.mobilenet_v3_small(pretrained=False)

        in_feats = backbone.classifier[0].in_features if isinstance(backbone.classifier[0], nn.Linear) else backbone.classifier[3].in_features
        # Replace classifier
        if isinstance(backbone.classifier, nn.Sequential):
            backbone.classifier = nn.Sequential(
                nn.Linear(in_feats, 512),
                nn.Hardswish(),
                nn.Dropout(0.2),
                nn.Linear(512, num_classes)
            )
        else:
            backbone.classifier = nn.Linear(in_feats, num_classes)
        self.model = backbone

    def forward(self, x):
        return self.model(x)

    def freeze_backbone(self, freeze: bool = True):
        # Keep classifier trainable; freeze features
        for name, p in self.model.named_parameters():
            if "classifier" in name:
                p.requires_grad = True
            else:
                p.requires_grad = not (not "classifier" in name and freeze)
