from enum import Enum, auto

class BrainMode(Enum):
    """
    Enum representing the different modes of the brain.
    """
    BYPASS = auto()
    NORMAL = auto()
    TRAINING = auto()