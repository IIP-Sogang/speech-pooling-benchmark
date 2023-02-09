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
        self.pooling = select_method(head_type, **kwargs)
        self.linear = SimpleLinear(input_dim, input_dim)
        self.fc = nn.Linear(input_dim, num_classes, bias=False) # AM-Softmax

    def forward(self, inputs:Tensor) -> Tensor:
        if self.training:
            aggregated = self.pooling(inputs)
            sv_embedding = self.linear(aggregated) # 768

            # For additive margin softmax loss
            for W in self.fc.parameters():
                W = F.normalize(W, dim=1)
            sv_embedding = F.normalize(sv_embedding, dim=1)
            outputs = self.fc(sv_embedding)
            return outputs # 1211
        else:
            return self.predict(inputs)

    def predict(self, inputs:Tensor) -> Tensor:
        assert inputs.size(0) == 2, "Two Speakers are needed for Verification Task!!"
        aggregated_0 = self.pooling(inputs[0])
        sv_embedding_0 = self.linear(aggregated_0)
        
        aggregated_1 = self.pooling(inputs[1])
        sv_embedding_1 = self.linear(aggregated_1)

        outputs = F.cosine_similarity(sv_embedding_0, sv_embedding_1, dim=1)
        return outputs
            