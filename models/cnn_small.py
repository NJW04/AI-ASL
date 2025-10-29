"""Configurable CNN for ASL classification with selectable activation and depth."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation(name: str):
    """Return an activation module by name.

    Parameters
    ----------
    name : str
        Activation identifier, e.g. ``"relu"`` or ``"gelu"``.

    Returns
    -------
    torch.nn.Module
        Instantiated activation module.

    Raises
    ------
    ValueError
        If the activation name is unknown.
    """
    if name == "relu":
        return nn.ReLU(inplace=True)
    elif name == "gelu":
        return nn.GELU()
    else:
        raise ValueError(f"Unknown activation function: {name}")


class ConvBlock(nn.Module):
    """Two 3×3 conv layers with BN+activation followed by 2×2 max-pooling."""

    def __init__(self, in_ch, out_ch, activation_fn: nn.Module):
        """Initialize the convolutional block.

        Parameters
        ----------
        in_ch : int
            Number of input channels.
        out_ch : int
            Number of output channels.
        activation_fn : nn.Module
            Activation module to insert between conv/BN layers.
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            activation_fn,
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            activation_fn,
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        """Apply the block to ``x``."""
        return self.conv(x)


class CNNSmall(nn.Module):
    """Compact CNN with configurable blocks and activation.

    Architecture
    -----------
    Repeats ``ConvBlock`` (channels double each block), then:
    GlobalAvgPool → Dropout → Linear(num_classes).
    """

    def __init__(
        self,
        num_classes: int = 29,
        base_channels: int = 32,
        dropout: float = 0.3,
        num_blocks: int = 3,
        activation: str = "relu",
    ):
        """Initialize the CNN.

        Parameters
        ----------
        num_classes : int
            Number of output classes.
        base_channels : int
            Channels of the first block; doubles each block.
        dropout : float
            Dropout probability before the final linear layer.
        num_blocks : int
            Number of ConvBlocks.
        activation : str
            Activation name passed to :func:`get_activation`.
        """
        super().__init__()

        activation_fn = get_activation(activation)

        self.blocks = nn.ModuleList()
        in_channels = 3
        current_channels = base_channels
        for i in range(num_blocks):
            self.blocks.append(ConvBlock(in_channels, current_channels, activation_fn))
            in_channels = current_channels
            current_channels *= 2

        final_channels = base_channels * (2 ** (num_blocks - 1))

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(final_channels, num_classes)

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nonlinearity_param = activation if activation in ["relu", "leaky_relu"] else "relu"
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity=nonlinearity_param)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        """Compute logits for input batch ``x``."""
        for block in self.blocks:
            x = block(x)
        x = self.gap(x).flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
