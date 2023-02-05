from typing import Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from models.head import select_head
from models.modules import SimpleLinear
from models.tasks.absract import TaskDependentModule

from x_vector import X_vector


class SpeakerVerificationModule(TaskDependentModule):
    def __init__(self, input_dim:int = 768, head_type='avgpool', **kwargs) -> None:
        super().__init__()
        self.x_vector_extractor = X_vector(input_dim=input_dim)
        self.head = select_head(head_type, **kwargs)
        self.linear = SimpleLinear(512, input_dim)

    def forward(self, inputs: Tensor) -> Tensor:
        x_vector = self.x_vector_extractor(inputs)
        speech_representation = self.head(x_vector)
        outputs = self.linear(speech_representation)
        return outputs

    def predict(self, inputs) -> Union[int, Tensor]:
        outputs = self.forward(inputs)
        # Output끼리 비교, 같은 쌍 구해서 출력하기
        return outputs.max(-1)