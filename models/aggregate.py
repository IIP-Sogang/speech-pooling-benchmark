from abc import *
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


class AvgPool(nn.Module):
    
    def forward(self, input_feature:Tensor, input_lengths:Tensor, *args, **kwargs):
        """
        Input feature size should follow (Batch size, n_layers, Length, Dimension)
        Return speech representation which follows (Batch size, Dimension)
        """
        assert input_feature.dim() == 4, f"Input feature size is {input_feature.size()}, Should follows (Batch, Layer, Length, Dimension)"
        outputs = input_feature.sum(2) / input_lengths[:,None,None]
        return outputs


class SimpleAvgPool(nn.Module):
    """
    This class utilizes only the last layer of 12-layered features.
    """

    def forward(self, input_feature:Tensor, input_lengths:Tensor, *args, **kwargs):
        """
        Input feature size should follow (Batch size, n_layers, Length, Dimension)
        Return speech representation which follows (Batch size, Dimension)
        """
        assert input_feature.dim() == 4, f"Input feature size is {input_feature.size()}, Should follows (Batch, Layer, Length, Dimension)"
        input_feature = input_feature[:,-1:]
        outputs = AvgPool()(input_feature, input_lengths)
        return outputs[:,0]


class VQWeightedAvgPool(nn.Module):
    """
    Average by VQ indices.
    """
    def __init__(self, shrink='exact') -> None:
        super().__init__()
        self._eq_key = shrink
    
    def forward(self, input_feature:Tensor, input_lengths:Tensor, vq_indices:Tensor):
        """
        Input feature size should follow (Batch size, n_layers, Length, Dimension)
        vq index size should follow (Batch size, Length, 2)
        Return speech representation which follows (Batch size, Dimension)
        """
        from itertools import groupby
        assert input_feature.dim() == 4, f"Input feature size is {input_feature.size()}, Should follows (Batch, Layer, Length, Dimension)"
        input_feature = input_feature[:,-1:] # Select last layer only
        B, N, L, D = input_feature.shape

        vq_indices = vq_indices.tolist() # This enormously accelarates the groupby calculation.
        avg_weights = torch.zeros((B, 1, L, 1), device=input_feature.device)
        for i, (vq_indices_, input_length) in enumerate(zip(vq_indices, input_lengths)):
            lengths = torch.tensor(
                [len(list(group)) for eq_value, group in groupby(vq_indices_[:input_length], key=self.eq_key)]
            ).to(input_feature.device)

            # ex) lengths = [1, 1, 3, 1] means there are 4 unique items, 
            # and items from 3rd to 5th are equal (3 repeated items).
            weightList = 1 / ( lengths.size(0) * lengths )
            # ex) weightList = [0.25, 0.25, 0.08, 0.25]
            # We reduce redundant items, utilizing unique items more
            avg_weights[i, :, :input_length] = torch.repeat_interleave(weightList, lengths)[:,None]
            # ex) avg_weights[0, 0] = [0.25, 0.25, 0.08, 0.08, 0.08, 0.25, 0, 0, 0, 0]
            # Match averaging weights to the input items
        
        outputs = (input_feature * avg_weights).sum(2) # Length dimension
        outputs = outputs[:, -1, :] # Squeeze dimension
        return outputs

    def eq_key(self, x):
        # You can define your own key, for different matching.
        # Refer to https://docs.python.org/3/library/itertools.html#itertools.groupby
        if self._eq_key=='exact':
            return tuple(x)
        elif self._eq_key=='former':
            return x[0]
        elif self._eq_key=='later':
            return x[-1]
        else:
            Exception


class SelfAttentivePooling(nn.Module):
    def __init__(self, input_dim:int = 768):
        super().__init__()

        self.sap_linear = nn.Linear(input_dim, input_dim)
        self.attention = self.new_parameter(input_dim, 1)

    def new_parameter(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out

    def forward(self, input_feature:Tensor, input_lengths:Tensor, *args):
        """
        Input feature size should follow (Batch size, n_layers, Length, Dimension)
        Return speech representation which follows (Batch size, Dimension)
        """
        import pdb;pdb.set_trace()
        assert input_feature.dim() == 4, f"Input feature size is {input_feature.size()}, Should follows (Batch, Layer, Length, Dimension)"
        input_feature = input_feature[:,-1] if input_feature.dim() == 4 else input_feature

        batch_size, feat_len, _ = input_feature.shape # (Batch size, Length, Dimension)

        h = torch.tanh(self.sap_linear(input_feature)) # (Batch size, Length, Dimension)
        w = torch.matmul(h, self.attention).squeeze(dim=2) # (Batch size, Length)
        w = F.softmax(w, dim=1).view(batch_size, feat_len, 1) # 
        feature = torch.sum(input_feature * w, dim=1) 

        return feature

class SelfAttentiveMaskingPooling(nn.Module):
    def __init__(self, input_dim:int = 768):
        super().__init__()

        self.sap_linear = nn.Linear(input_dim, input_dim)
        self.attention = self.new_parameter(input_dim, 1)

    def new_parameter(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out

    def forward(self, input_feature:Tensor, input_lengths:Tensor, *args):
        """
        Input feature size should follow (Batch size, n_layers, Length, Dimension)
        Return speech representation which follows (Batch size, Dimension)
        """
        import pdb;pdb.set_trace()
        assert input_feature.dim() == 4, f"Input feature size is {input_feature.size()}, Should follows (Batch, Layer, Length, Dimension)"
        input_feature = input_feature[:,-1] if input_feature.dim() == 4 else input_feature

        batch_size, feat_len, _ = input_feature.shape # (Batch size, Length, Dimension)

        h = torch.tanh(self.sap_linear(input_feature)) # (Batch size, Length, Dimension)
        w = torch.matmul(h, self.attention).squeeze(dim=2) # (Batch size, Length)

        # If length = 2, mask = [[1, 1, 0, 0, 0, ...]]
        mask = torch.arange(feat_len).expand(batch_size, feat_len).to(w.device) < input_lengths
        w = w + (~mask) * (w.min() - 20)

        w = F.softmax(w, dim=1).view(batch_size, feat_len, 1) # 
        feature = torch.sum(input_feature * w, dim=1) 

        return feature


class WhiteningBERT(nn.Module):
    def __init__(self, layer_ids:List[int]=None, whitening=True):
        super().__init__()
        print("layers", layer_ids)
        print("whitening", whitening)
        self.pool = AvgPool()
        self.layer_comb = LayerCombination(layer_ids=layer_ids)
        self.whitening = Whitening() if whitening else nn.Identity()

    def forward(self, input_feature:Tensor, input_lengths:Tensor, *args):
        """
        Input feature size should follow (Batch size, n_layers, Length, Dimension)
        Return speech representation which follows (Batch size, Dimension)
        """
        pool_feature = self.pool(input_feature, input_lengths)
        combined_feature = self.layer_comb(pool_feature)
        whiten_feature = self.whitening(combined_feature)
        return whiten_feature


class LayerCombination(nn.Module):
    def __init__(self, layer_ids:List[int]=None):
        super().__init__()
        self.layer_ids = layer_ids

    def forward(self, input_feature:Tensor):
        """
        Input feature size should follow (Batch size, n_layers, Length, Dimension)
        Return speech representation which follows (Batch size, Length, Dimension)
        """
        combined_feature = input_feature[:,self.layer_ids].mean(1)
        return combined_feature


class Whitening(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, input_feature:Tensor):
        """
        Input feature size should follow (Batch size, Dimension)
        Return speech representation which follows (Batch size, Dimension)
        """
        m = input_feature.mean(dim=0, keepdim=True)
        Cov = torch.mm((input_feature-m).T,(input_feature-m))
        # eigval, eigvec = torch.linalg.eig(Cov)
        # D = torch.diag(eigval ** -0.5)
        # whiten_feature = (input_feature - m) @ eigvec @ D
        U, S, V = torch.svd(Cov)
        D = torch.diag(1/torch.sqrt(S))
        whiten_feature = torch.mm(input_feature - m, torch.mm(U,D))
        return whiten_feature


def select_method(head_type:str='avgpool', input_dim:int=768, layer_ids:Union[str,List[int],int]="1 12", **kwargs):
    if isinstance(layer_ids, str):
        layer_ids = list(map(int, layer_ids.split()))
    elif isinstance(layer_ids, int):
        layer_ids = [layer_ids]
    elif isinstance(layer_ids, list):
        layer_ids = layer_ids

    if head_type=='avgpool':
        return SimpleAvgPool()
    elif head_type=='sap':
        return SelfAttentivePooling(input_dim)
    elif head_type=='white':
        return WhiteningBERT(layer_ids=layer_ids, whitening=kwargs['whitening'])
    elif head_type=='vq':
        return VQWeightedAvgPool(shrink=kwargs['shrink'])
    else:
        assert False, f"""HEAD TYPE "{head_type}" IS NOT IMPLEMENTED!"""