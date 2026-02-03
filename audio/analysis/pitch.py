import numpy as np
import librosa 

# Estimate the funamental frequency (f0) of a signal
def estimate_f0(
        audio,
        sr,
        fmin=30.0,
        fmax=2000.0,
):
    # Use librosa's pyin algorithm to estimate f0
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
    )

    # Extract only voiced f0 values
    f0_voiced = f0[~np.isnan(f0)]

    # If no voiced segments found, return None
    if len(f0_voiced) == 0:
        return None
    

    # Return median F0 of voiced segments (to account for any detune)
    return float(np.median(f0_voiced))