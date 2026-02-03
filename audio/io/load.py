from pathlib import Path
import numpy as np
import soundfile as sf

def load_wav(
        path,
        target_sr=44100,
        mono=True,
        dtype=np.float32
):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    audio, sr = sf.read(path, always_2d=True)

    if mono:
        audio = np.mean(audio, axis=1)

    if sr != target_sr:
        raise ValueError(f"Sample rate mismatch: expected {target_sr}, got {sr}")
    
    audio = audio.astype(dtype)

    return audio, sr
