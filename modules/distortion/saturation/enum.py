from enum import Enum

class SaturationType(Enum):
    TANH = "tanh"
    ATAN = "atan"
    HARD_CLIP = "hard_clip"
    POLY = "poly"
    ASYMMETRIC = "asymmetric"