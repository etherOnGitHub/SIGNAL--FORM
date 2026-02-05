from enum import Enum

class WindowType(Enum):
    """
    Enum representing different types of window functions for FFT analysis.
    """
    HANN = "hann"
    HAMMING = "hamming"
    BLACKMAN = "blackman"
    RECT = "rectangular"