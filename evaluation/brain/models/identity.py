import torch
import torch.nn as nn

class IdentityModel(nn.Module):
    def foward(self, x: torch.Tensor) -> torch.Tensor:
        return x