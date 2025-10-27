#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# Helper function to get activation module
def get_activation(name: str):
    if name == "relu":
        return nn.ReLU(inplace=True)
    elif name == "gelu":
        return nn.GELU()
    # Add other activations here if needed (e.g., nn.SiLU())
    else:
        raise ValueError(f"Unknown activation function: {name}")

class ConvBlock(nn.Module):
    # Added activation_fn argument
    def __init__(self, in_ch, out_ch, activation_fn: nn.Module):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            activation_fn, # Use passed activation
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            activation_fn, # Use passed activation
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.conv(x)


class CNNSmall(nn.Module):
    """
    Compact CNN with configurable blocks and activation:
      3x3 convs with BN+Activation, MaxPool; channels: c->2c->4c->...
      GlobalAvgPool -> Dropout -> Linear(num_classes)
    """
    # Added num_blocks and activation arguments
    def __init__(self, num_classes: int = 29, base_channels: int = 32, dropout: float = 0.3, 
                 num_blocks: int = 3, activation: str = "relu"):
        super().__init__()
        
        activation_fn = get_activation(activation) # Get the activation module
        
        # Build blocks dynamically
        self.blocks = nn.ModuleList()
        in_channels = 3
        current_channels = base_channels
        for i in range(num_blocks):
            self.blocks.append(ConvBlock(in_channels, current_channels, activation_fn))
            in_channels = current_channels
            current_channels *= 2 # Double channels for next block
            
        final_channels = base_channels * (2**(num_blocks - 1)) # Channels out of the last block

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(final_channels, num_classes)

        # He init (or appropriate init based on activation)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Use nonlinearity parameter matching the chosen activation
                nonlinearity_param = activation if activation in ['relu', 'leaky_relu'] else 'relu' # Default for kaiming if activation unknown to it
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity=nonlinearity_param)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        # Pass through all blocks
        for block in self.blocks:
            x = block(x)
            
        x = self.gap(x).flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        return x