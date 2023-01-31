from abc import *
from typing import Optional, Tuple, Union

import torch
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


class KeywordSpottingModule(TaskDependentModule):
    def __init__(self, input_dim:int = 768, num_classes:int = 30, head_type='avgpool') -> None:
        super().__init__()
        self.linear = SimpleLinear(input_dim, num_classes)
        self.head = select_head(head_type)

    def forward(self, inputs) -> Tensor:
        speech_representation = self.head(inputs)
        outputs = self.linear(speech_representation)
        return outputs

    def predict(self, inputs) -> Union[int, Tensor]:
        speech_representation = self.head(inputs)
        outputs = self.linear(speech_representation)
        return outputs.max(-1)


class SimpleAvgPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def forward(self, input_feature:Tensor):
        """
        Input feature size should follow (Batch size, Length, Dimension)
        Return speech representation which follows (Batch size, Dimension)
        """
        return self.avgpool(input_feature.permute(0,2,1)).squeeze(-1)


class SimpleLinear(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.linear = nn.Linear(*args, **kwargs)

    def forward(self, represents:Tensor):
        logits = self.linear(represents)
        return logits


def select_head(head_type:str='avgpool'):
    if head_type=='avgpool':
        return SimpleAvgPool()
    else:
        assert False, f"""HEAD TYPE "{head_type}" IS NOT IMPLEMENTED!"""