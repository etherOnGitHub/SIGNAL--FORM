"""
Audio context data structure for holding analysis results.
This module defines the AudioContext class which encapsulates various audio analysis results.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import numpy as np

@dataclass
class AudioContext:
    sample_rate: int
    f0: Optional[float] = None
    loudness: Optional[float] = None
    spectrogram: Optional[np.ndarray] = None
    features: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "f0": self.f0,
            "sample_rate": self.sample_rate,
            "loudness": self.loudness,
            "spectrogram": self.spectrogram is not None,
            "features": list(self.features.keys()),
        }