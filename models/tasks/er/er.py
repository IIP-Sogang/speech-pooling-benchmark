from typing import Union

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from models.aggregate import select_method
from models.modules import SimpleLinear
from models.tasks.absract import TaskDependentModule


class EmotionRecognitionModule(TaskDependentModule):
    def __init__(self, input_dim:int = 768, num_classes:int = 30, head_type='avgpool', **kwargs) -> None:
        super().__init__()
        self.head = select_head(head_type, **kwargs)
        self.linear = SimpleLinear(input_dim, num_classes)

    def forward(self, inputs) -> Tensor:
        speech_representation = self.head(inputs)
        outputs = self.linear(speech_representation)
        return outputs

    def predict(self, inputs) -> Union[int, Tensor]:
        speech_representation = self.head(inputs)
        outputs = self.linear(speech_representation)
        return outputs.max(-1)