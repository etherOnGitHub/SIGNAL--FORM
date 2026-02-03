"""
Blend two audio signals based on a mix parameter.
"""

import numpy as np

# Function to blend dry and wet signals
def linear_blend(
        dry: np.ndarray,
        wet: np.ndarray,
        mix: float,
        ) -> np.ndarray:
    # If mix is 0.0, return dry signal or if 1.0, return wet signal
    if mix <= 0.0:
        return dry
    
    if mix >= 1.0:
        return wet
    
    return (1.0 - mix) * dry + mix * wet