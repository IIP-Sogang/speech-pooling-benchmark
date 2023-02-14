from typing import Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from models.aggregate import select_method
from models.modules import SimpleLinear
from models.tasks.absract import TaskDependentModule


class SpeakerVerificationModule(TaskDependentModule):
    def __init__(self, input_dim:int = 768, num_classes:int = 1211, embedding_size:int = 512, head_type='avgpool', **kwargs) -> None:
        super().__init__()
        self.pooling = select_method(head_type, **kwargs)
        self.linear = SimpleLinear(input_dim, embedding_size)
        self.fc = nn.Linear(embedding_size, num_classes, bias=False) # AM-Softmax

    def forward(self, inputs:Tensor, input_lengths, vq_indices=None) -> Tensor:
        if self.training:
            aggregated = self.pooling(inputs, input_lengths, vq_indices=vq_indices)
            sv_embedding = self.linear(aggregated) # 768

            # For additive margin softmax loss
            for W in self.fc.parameters():
                W = F.normalize(W, dim=1)
            sv_embedding = F.normalize(sv_embedding, dim=1)
            outputs = self.fc(sv_embedding)
            return outputs # 1211
        else:
            return self.predict(inputs, input_lengths, vq_indices=vq_indices)

    def predict(self, inputs:Tensor, input_lengths, vq_indices=None) -> Tensor:
        assert inputs.size(0) == 2, "Two Speakers are needed for Verification Task!!"
        aggregated_0 = self.pooling(inputs[0], input_lengths[0], vq_indices=vq_indices[0])
        sv_embedding_0 = self.linear(aggregated_0)
        
        aggregated_1 = self.pooling(inputs[1], input_lengths[1], vq_indices=vq_indices[1])
        sv_embedding_1 = self.linear(aggregated_1)

        outputs = F.cosine_similarity(sv_embedding_0, sv_embedding_1, dim=1)
        return outputs
            