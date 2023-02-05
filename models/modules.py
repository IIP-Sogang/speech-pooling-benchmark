import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

class SimpleLinear(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.linear = nn.Linear(*args, **kwargs)

    def forward(self, represents:Tensor):
        logits = self.linear(represents)
        return logits