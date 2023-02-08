from typing import Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from models.aggregate import select_method
from models.modules import SimpleLinear
from models.tasks.absract import TaskDependentModule


class SpeakerVerificationModule(TaskDependentModule):
    def __init__(self, input_dim:int = 768, num_classes:int = 30, head_type='avgpool', **kwargs) -> None:
        super().__init__()
        self.head = select_method(head_type, **kwargs)
        self.linear = SimpleLinear(input_dim, input_dim)
        self.fc = nn.Linear(input_dim, num_classes, bias=False)

    def forward(self, inputs: Tensor) -> Tensor:
        aggregated = self.head(inputs)
        sv_embedding = self.linear(aggregated)

        # For additive margin softmax loss
        for W in self.fc.parameters():
            W = F.normalize(W, dim=1)
        sv_embedding = F.normalize(sv_embedding, dim=1)
        outputs = self.fc(sv_embedding)
        return outputs

    def predict(self, inputs) -> Union[int, Tensor]:
        # NOT IMPLEMENTED YET
        sv_embeddings = self.linear(self.head(inputs))
        # Output끼리 비교, 같은 쌍 구해서 출력하기
        outputs = None
        return outputs.max(-1)