from abc import *
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


class SimpleAvgPool(nn.Module):
    def __init__(self):
        super().__init__()
        # self.avgpool = nn.AdaptiveAvgPool1d(1)

    def forward(self, input_feature:Tensor):
        """
        Input feature size should follow (Batch size, Length, Dimension) or (Batch size, n_layers, Length, Dimension)
        Return speech representation which follows (Batch size, Dimension)
        """
        if input_feature.dim() == 3:
            # return self.avgpool(input_feature.permute(0,2,1)).squeeze(-1)
            return input_feature.mean(1)
        elif input_feature.dim() == 4:
            return input_feature.mean(2)
        else:
            raise Exception

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
    def __init__(self, layer_ids:Union[str, int, List[int]]=None, whitening=True):
        super().__init__()
        print("layers", layer_ids)
        print("whitening", whitening)
        self.pool = SimpleAvgPool()
        self.layer_comb = LayerCombination(layer_ids=layer_ids)
        self.whitening = Whitening() if whitening else nn.Identity()

    def forward(self, input_feature:Tensor):
        """
        Input feature size should follow (Batch size, n_layers, Length, Dimension)
        Return speech representation which follows (Batch size, Dimension)
        """
        pool_feature = self.pool(input_feature)
        combined_feature = self.layer_comb(pool_feature)
        whiten_feature = self.whitening(combined_feature)
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
        import pdb; pdb.set_trace()
        m = input_feature.mean(dim=0, keepdim=True)
        Cov = torch.mm((input_feature-m).T,(input_feature-m))
        # eigval, eigvec = torch.linalg.eig(Cov)
        # D = torch.diag(eigval ** -0.5)
        # whiten_feature = (input_feature - m) @ eigvec @ D
        U, S, V = torch.svd(Cov)
        D = torch.diag(1/torch.sqrt(S))
        whiten_feature = torch.mm(input_feature - m, torch.mm(U,D))
        return whiten_feature


def select_head(head_type:str='avgpool', 
                input_dim:int=768, layer_ids:Union[str,List[int],int]="1 12", whitening:bool=True, **kwargs):
    if head_type=='avgpool':
        return SimpleAvgPool()
    elif head_type=='sap':
        return SelfAttentivePooling(input_dim)
    elif head_type=='white':
        return WhiteningBERT(layer_ids=layer_ids, whitening=whitening)
    else:
        assert False, f"""HEAD TYPE "{head_type}" IS NOT IMPLEMENTED!"""