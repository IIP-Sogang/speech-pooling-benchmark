from abc import *
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


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

class SelfAttentivePooling(nn.Module):
    def __init__(self, input_dim:int = 768):
        super().__init__()

        self.sap_linear = nn.Linear(input_dim, input_dim)
        self.attention = self.new_parameter(input_dim, 1)

    def new_parameter(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out

    def forward(self, input_feature:Tensor):
        """
        Input feature size should follow (Batch size, Length, Dimension)
        Return speech representation which follows (Batch size, Dimension)
        """
        h = torch.tanh(self.sap_linear(input_feature)) # 
        w = torch.matmul(h, self.attention).squeeze(dim=2)
        w = F.softmax(w, dim=1).view(input_feature.size(0), input_feature.size(1), 1) # 
        feature = torch.sum(input_feature * w, dim=1) 

        return feature


class WhiteningBERT(nn.Module):
    def __init__(self, layer_ids:Union[str, int, List[int]]=None):
        super().__init__()
        self.pool = SimpleAvgPool()
        self.layer_comb = LayerCombination(layer_ids=layer_ids)
        self.whitening = Whitening()

    def forward(self, input_feature:Tensor):
        """
        Input feature size should follow (Batch size, n_layers, Length, Dimension)
        Return speech representation which follows (Batch size, Dimension)
        """
        combined_feature = self.layer_comb(input_feature)
        pool_feature = self.pool(combined_feature)
        whiten_feature = self.whitening(pool_feature)
        return whiten_feature


class LayerCombination(nn.Module):
    def __init__(self, layer_ids:Union[str, int, List[int]]=None):
        super().__init__()
        if isinstance(layer_ids, str):
            self.layer_ids = list(map(int, layer_ids.split()))
        elif isinstance(layer_ids, int):
            self.layer_ids = [layer_ids]
        elif isinstance(layer_ids, list):
            self.layer_ids = layer_ids

    def forward(self, input_feature:Tensor):
        """
        Input feature size should follow (Batch size, n_layers, Length, Dimension)
        Return speech representation which follows (Batch size, Length, Dimension)
        """
        combinated_vector = torch.zeros_like(input_feature[:,0])
        for layer_id in self.layer_ids:
            combinated_vector += input_feature[:, layer_id]
        combinated_feature = combinated_vector / len(self.layer_ids)
        return combinated_feature


class Whitening(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, input_feature:Tensor):
        """
        Input feature size should follow (Batch size, Length, Dimension)
        Return speech representation which follows (Batch size, Dimension)
        """
        m = input_feature.mean(dim=0)
        Cov = (input_feature-m).T @ (input_feature-m)
        svd = torch.svd(Cov)
        U = svd.U
        S = svd.S
        D = torch.diag(S ** -0.5)
        whiten_feature = (input_feature - m) @ U @ D
        return whiten_feature


class SimpleLinear(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.linear = nn.Linear(*args, **kwargs)

    def forward(self, represents:Tensor):
        logits = self.linear(represents)
        return logits


def select_head(head_type:str='avgpool', 
                input_dim:int=768, layer_ids:Union[str,List[int],int]="1 12", **kwargs):
    if head_type=='avgpool':
        return SimpleAvgPool()
    elif head_type=='sap':
        return SelfAttentivePooling(input_dim)
    elif head_type=='white':
        return WhiteningBERT(layer_ids=layer_ids)
    else:
        assert False, f"""HEAD TYPE "{head_type}" IS NOT IMPLEMENTED!"""