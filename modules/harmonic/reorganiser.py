from __future__ import annotations

import torch
import torch.nn as nn

class HarmonicReorganiser(nn.Module):
    def __init__(self):
        super().__init__()

        # No learnable parameters, so we don't need to define any layers

        pass

    def forward(self, x_ri: torch.Tensor) -> torch.Tensor:
        # Check input shape
        if x_ri.ndim != 4 or x_ri.shape[1] != 2:
            raise ValueError("Input tensor must have shape (batch, 2, freq_bins, time_frames)")
        """
        Indentity masking layer that simply returns the input as output. 1 + 0j
        """
        return x_ri
        