from typing import Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from models.aggregate import select_head
from models.modules import SimpleLinear
from models.tasks.absract import TaskDependentModule


class SpeakerVerificationModule(TaskDependentModule):
    def __init__(self, input_dim:int = 768, head_type='avgpool', **kwargs) -> None:
        super().__init__()
        self.head = select_head(head_type, **kwargs)
        self.linear = SimpleLinear(512, input_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        speech_representation = self.head(inputs)
        outputs = self.linear(speech_representation)
        return outputs

    def predict(self, inputs) -> Union[int, Tensor]:
        outputs = self.forward(inputs)
        # Output끼리 비교, 같은 쌍 구해서 출력하기
        return outputs.max(-1)