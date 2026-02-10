import torch
from torch import Tensor
from evaluation.brain.models.base import HarmonicusBase

class IdentityBrain(HarmonicusBase):
    def __init__(self):
        super().__init__()

    def forward(
        self, 
        x_ri: Tensor, 
        context: object | None = None
    ) -> Tensor:
        return x_ri