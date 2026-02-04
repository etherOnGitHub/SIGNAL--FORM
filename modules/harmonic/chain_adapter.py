from __future__ import annotations

import numpy as np
import torch

from modules.base.module import BaseModule
from audio.analysis.stft import STFTConfig, stft_ri, istft_from_ri

"""
Numpy to Torch adapter for HarmonicReorganiser module.
We need to convert numpy arrays to torch tensors and vice versa.
DSP CHAIN -- Numpy  - (T,)
HarmonicReorganiser -- Torch - (B, 2, F, T)
"""

class HarmonicToChainAdapter(BaseModule):
    def __init__(
            self,
            model: torch.nn.Module,
            stft_cfg: STFTConfig,
            device: str | None = None,
        ):
        # Initialize base module
        super().__init__()
        self.model = model
        self.stft_cfg = stft_cfg
        self.device = torch.device(device) if device else torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()
    # Process method to handle numpy input and output
    def process(self, audio: np.ndarray, context) -> np.ndarray:
        # Validate input
        if not isinstance(audio, np.ndarray):
            raise TypeError("Input audio must be a numpy array")
        
        if audio.ndim != 1:
            raise ValueError("Input audio numpy array must be 1D")

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32, copy=False)
        # Remember original length for iSTFT
        length = audio.shape[0]
        # Convert numpy array to torch tensor
        x = torch.from_numpy(audio).to(self.device)
        # Compute STFT (real and imaginary parts)
        X_ri = stft_ri(x, self.stft_cfg)
        # Add batch dimension
        X_ri = X_ri.unsqueeze(0)
        # Forward pass through the model
        with torch.no_grad():
            Y_ri = self.model(X_ri)
        # Validate output shape
        if Y_ri.shape != X_ri.shape:
            raise RuntimeError("Output shape from model does not match input shape")
        # Convert output back to numpy array
        Y_ri = Y_ri.squeeze(0)  # Remove batch dimension
        # Compute iSTFT to get time-domain signal
        y = istft_from_ri(Y_ri, self.stft_cfg, length=length)
        # Convert to numpy array
        out = y.detach().cpu().numpy().astype(np.float32, copy=False)
        return out