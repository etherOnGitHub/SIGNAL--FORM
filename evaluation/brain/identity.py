from evaluation.brain.modes import BrainMode
from evaluation.brain.harmonicus import Harmonicus

class IdentityHarmonicus(Harmonicus):
        def __init__(self):
            super().__init__(
                model=None,
                mode=BrainMode.BYPASS,
            )