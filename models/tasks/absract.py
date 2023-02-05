from abc import *
from typing import List, Optional, Tuple, Union

import torch.nn as nn
from torch import Tensor

class TaskDependentModule(nn.Module, ABC):
    def __init__(self) -> None:
        super().__init__()
        pass

    @abstractmethod
    def forward(self, inputs) -> Tensor:
        pass

    @abstractmethod
    def predict(self, inputs) -> Union[int, Tensor]:
        pass