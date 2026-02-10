from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from evaluation.brain.models.base import HarmonicusBase

"""
A simple brain that predicts a mask from input and applies it to the input.
This is a simple model that can learn to suppress certain frequencies or time regions in the input.
Convolutional layers are used to capture local patterns in the input.
"""

class MaskBrain(HarmonicusBase):
    def __init__(self, channels: int = 32):
        super().__init__()

        # A simple convolutional network to predict a mask from the input
        self.net = nn.Sequential(
            nn.Conv2d(2, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, 1, kernel_size=1),
        )

        # Initialize the last layer to produce small values, so the initial mask is close to 1
    def forward(
        self, 
        x_ri: Tensor, 
        context: object | None = None
    ) -> Tensor:
        # Predict a mask from the input
        m = self.net(x_ri)
        # Use a non-linear activation to ensure the mask is positive and close to 1
        m = 1.0 + 2 * torch.exp(torch.tanh(m))
        m = torch.clamp(m, 0.5, 5.0) # limit the range of the mask to prevent extreme values
        return x_ri * m