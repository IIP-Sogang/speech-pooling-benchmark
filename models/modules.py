import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

class SimpleLinear(nn.Module):
    def __init__(self, input_dim, num_classes, **kwargs):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, represents:Tensor):
        logits = self.linear(represents)
        return logits