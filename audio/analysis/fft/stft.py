"""
STFT - Short-Time Fourier Transform module.
Provides functionality to compute and manipulate the STFT of audio signals.
We take apart a signal into its frequency components over time.
This can be then sent to a machine learning model to process or modify the audio in the frequency domain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
from .windows import WindowType

import torch

# Import STFT
@dataclass(frozen=True)
class STFTConfig:
    # bin count
    n_fft: int = 2048
    # length between frames
    hop_length: int = 256
    # window length (window = size of each frame)
    win_length: int | None = None
    window: WindowType = WindowType.HANN # type of window
    # whether to pad the input on both sides
    center: bool = True
    # padding mode
    pad_mode: str = "reflect"

# Helper to create window tensor
# NOTE: window recreation is intentional for now
# Can be cached per (cfg, device, dtype) later
def _make_window(cfg: STFTConfig, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    win_length = cfg.win_length or cfg.n_fft
    if cfg.window is WindowType.HANN:
        return torch.hann_window(win_length, periodic=True, device=device, dtype=dtype)
    else:
        raise ValueError(f"Unsupported window type: {cfg.window}")
"""
Compute the Short-Time Fourier Transform (STFT) of the input audio signal.

Args:
    audio (torch.Tensor): Input audio signal of shape (batch, time).
    cfg (STFTConfig): Configuration for STFT parameters.

Returns:
    torch.Tensor: Complex STFT of shape (batch, freq_bins, time_frames).
"""
@torch.no_grad()
def stft_complex(
    audio: torch.Tensor,
    cfg: STFTConfig
) -> torch.Tensor:
    # Check input dimensions
    if audio.ndim == 1:
        audio_in = audio.unsqueeze(0)
        squeeze_back = True
    elif audio.ndim == 2:
        audio_in = audio
        squeeze_back = False
    else:
        raise ValueError("Audio tensor must be 1D or 2D (batch, time)")
    
    device = audio_in.device
    dtype = audio_in.dtype
    window = _make_window(cfg, device, dtype)
    # Compute STFT
    X = torch.stft(
        audio_in,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length or cfg.n_fft,
        window=window,
        center=cfg.center,
        pad_mode=cfg.pad_mode,
        normalized=False,
        onesided=True,
        return_complex=True,
    )

    return X.squeeze(0) if squeeze_back else X


"""
Compute the STFT and return real and imaginary parts separately.
This is useful for models that require separate channels for real and imaginary components.
returns:
    - (2, F(freq_bins), T(time_frames)) for single input
    - (B (batch), 2, F(freq_bins), T(time_frames)) for batch input
"""
# gradient = when we want to use the STFT in a model and backpropagate through it, we need to ensure that the operations are differentiable.
@torch.no_grad()
def stft_ri(
    audio: torch.Tensor,
    cfg: STFTConfig
) -> torch.Tensor:
    X = stft_complex(audio, cfg)
    # Separate real and imaginary parts
    if X.ndim == 2:
        return torch.stack((X.real, X.imag), dim=0)
    else:
        return torch.stack((X.real, X.imag),dim=1)
    
"""
Compute the inverse Short-Time Fourier Transform (iSTFT) from complex STFT representation.
This reconstructs the time-domain audio signal from its frequency-domain representation.
"""
@torch.no_grad()
def istft_from_complex(
    X: torch.Tensor,
    cfg: STFTConfig,
    length: int | None = None # desired output length
) -> torch.Tensor:
    # Check input type
    if X.dtype not in (torch.complex64, torch.complex128):
        raise ValueError("Input tensor must be of complex nature")

    device = X.device
    float_dtype = torch.float32 if X.dtype == torch.complex64 else torch.float64 # match precision
    window = _make_window(cfg, device, float_dtype)
    # Compute iSTFT
    y = torch.istft(
        X,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length or cfg.n_fft,
        window=window,
        center=cfg.center,
        normalized=False,
        onesided=True,
        length=length,
    )
    return y
"""
Compute the inverse Short-Time Fourier Transform (iSTFT) from real-imaginary STFT representation.
This reconstructs the time-domain audio signal from its frequency-domain representation.
Supports both single and batch inputs.
"""
@torch.no_grad()
def istft_from_ri(
    X_ri: torch.Tensor,
    cfg: STFTConfig,
    length: int | None = None
) -> torch.Tensor:
    # Check input dimensions
    if X_ri.ndim == 3:
        # Single input: (2, freq_bins, time_frames)
        real, imag = X_ri[0], X_ri[1]
        # Combine real and imaginary parts into complex tensor
        Xc = torch.complex(real, imag)
        return istft_from_complex(Xc, cfg, length=length)
    if X_ri.ndim == 4:
        # Batch input: (batch, 2, freq_bins, time_frames)
        real, imag = X_ri[:,0], X_ri[:,1]
        # Combine real and imaginary parts into complex tensor
        Xc = torch.complex(real, imag)
        return istft_from_complex(Xc, cfg, length=length)
    raise ValueError("Input tensor must be 3D or 4D for real-imaginary representation")

    