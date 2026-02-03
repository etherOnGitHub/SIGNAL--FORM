"""
Pitch analysis only.
This module must never modify audio.
"""

import numpy as np
import librosa 

# Estimate the funamental frequency (f0) of a signal
def estimate_f0(
        audio,
        sr,
        fmin=22,
        fmax=None,
        frame_length=4096,
):
    # Use librosa's pyin algorithm to estimate f0
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio,
        fmin=fmin,
        fmax=fmax or sr // 2,
        sr=sr,
        frame_length=frame_length,
    )

    # Extract only voiced f0 values
    f0_voiced = f0[~np.isnan(f0)]

    # If no voiced segments found, return None
    if len(f0_voiced) == 0:
        return None
    

    # Return median F0 of voiced segments (to account for any detune)
    return float(np.median(f0_voiced))