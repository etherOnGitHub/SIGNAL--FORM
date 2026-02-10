from __future__ import annotations

import torch
from torch import Tensor
from abc import ABC, abstractmethod

class HarmonicusBase(torch.nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod    
    def forward(
        self, 
        x_ri: Tensor, 
        context: object | None = None
    ) -> Tensor:
        raise NotImplementedError
