from __future__ import annotations
import torch

from evaluation.brain.modes import BrainMode

"""
This module defines the Harmonicus class, which is a wrapper around a PyTorch model for harmonic processing.

Harmonicus can operate in different modes (BYPASS, NORMAL, TRAINING) and handles the conversion between the 
input and output formats required by the model.

The forward method takes in a complex spectrogram (x_ri) and a context object, processes it through the model 
based on the current mode, and returns the processed spectrogram.
"""

class Harmonicus:
    def __init__(        
        self,
        model: torch.nn.Module | None = None,
        mode: BrainMode = BrainMode.BYPASS,
        device: torch.device | None = None,
    ):
        self.model = model
        self.mode = mode
        self.device = device or torch.device("cpu") # Default to CPU if no device is specified.
        if self.model is not None:
            self.model.to(self.device)

    def forward(self, x_ri: torch.Tensor, context) -> torch.Tensor:
        if context is None:
            raise ValueError("Context cannot be None Harmonicus needs it to process the audio.")
        # Check input shape
        if self.mode is BrainMode.BYPASS or self.model is None:
            return x_ri
        # Validate input shape
        if self.mode is BrainMode.NORMAL:
            # Ensure model is in evaluation mode and disable gradient computation.
            self.model.eval()
            with torch.no_grad():
                y = self.model(x_ri)
            return self._validate_output(x_ri, y)
        # In training mode, we want to allow gradients to flow through the model for backpropagation.
        if self.mode is BrainMode.TRAINING:
            self.model.train()
            y = self.model(x_ri)
            return self._validate_output(x_ri, y)
        
        raise RuntimeError(f"Unsupported mode: {self.mode}")
    
    # Private method to validate the model's output
    def _validate_output(
            self,
            x: torch.Tensor, 
            y: torch.Tensor
        ) -> torch.Tensor:
            if not torch.is_floating_point(y):
                raise RuntimeError("Output tensor must be a floating point tensor")
            if y.shape != x.shape:
                raise RuntimeError(f"Output shape {y.shape} does not match input shape {x.shape}")
            if torch.isnan(y).any():
                raise RuntimeError("Output tensor contains NaN values")
            return y 