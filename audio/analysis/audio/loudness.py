"""
Loudness analysis only.
This module must never modify audio.
"""

import numpy as np

# Estimate the loudness of a signal in dB using RMS - EPS floor for numerical stability
def estimate_loudness_db(audio, eps=1e-9):
    # Compute RMS
    rms = np.sqrt(np.mean(audio**2))

    # Convert to dB
    loudness_db = 20 * np.log10(rms + eps)

    return float(loudness_db)