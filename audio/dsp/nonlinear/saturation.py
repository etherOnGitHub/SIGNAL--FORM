"""
DSP Non-linear Saturation Functions
This module provides various non-linear saturation functions for audio signal processing.
--------------------------
TANH Saturation Function

Implements hyperbolic tangent saturation.
Math: y = tanh(gain * x)
--------------------------
ATAN Saturation Function

Implements arctangent saturation.
Math: y = (2 / π) * arctan(gain * x)
--------------------------
HARD CLIP Saturation Function

Implements hard clipping saturation.
Math: y = clip(gain * x, -1.0, 1.0)
--------------------------
POLY Saturation Function

Implements polynomial saturation.
Math: y = gain * x - (gain * x)^3 / 3
--------------------------
ASYMMETRIC Saturation Function

Implements asymmetric saturation.
Math: 
    y = tanh(gain * x) for x >= 0
    y = 0.5 * tanh(gain * x) for x < 0
--------------------------

"""

import numpy as np

def tanh_shaper(x: np.ndarray, gain: float) -> np.ndarray:
    return np.tanh(gain * x)

def atan_shaper(x: np.ndarray, gain: float) -> np.ndarray:
    return (2 / np.pi) * np.arctan(gain * x)

def hard_clip_shaper(x: np.ndarray, gain: float) -> np.ndarray:
    return np.clip(gain * x, -1.0, 1.0)

def poly_shaper(x: np.ndarray, gain: float) -> np.ndarray:
    y = gain * x
    return y - (y ** 3) / 3.0

def asymmetric_shaper(x: np.ndarray, gain: float) -> np.ndarray:
    y = gain * x
    return np.where(
        y >= 0.0, 
        np.tanh(y),
        0.5 * np.tanh(y)
        )