from abc import *
from typing import List, Optional, Tuple, Union
from itertools import groupby

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
    def __init__(self, shrink='exact', margin=0) -> None:
        super().__init__()
        self._eq_key = shrink
        self.margin = margin
    
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
    
    def extract_stats(self, input_feature:Tensor, input_length:Tensor, vq_index:Tensor):# ->List[Dict[str,int], List[int], List[int]]:
        """
        Input feature size should follow (n_layers, Length, Dimension)
        vq index size should follow (Length, 2)
        Return (vq_representation, token dictionary, cluster lengths, token length)
        """
        from itertools import groupby
        assert input_feature.dim() == 3, f"Input feature size is {input_feature.size()}, Should follows (Layer, Length, Dimension)"
        input_feature = input_feature[-1:] # Select last layer only
        N, L, D = input_feature.shape

        token_dict = dict()
        cluster_len = dict()

        vq_index = vq_index.tolist() # This enormously accelarates the groupby calculation.
        for vq_index_ in vq_index:
            # === Count index number ===
            token_dict[tuple(vq_index_)] = token_dict.get(tuple(vq_index_), 0) + 1

        avg_weights = torch.zeros((1, L, 1), device=input_feature.device)
        
        length = torch.tensor(
            [len(list(group)) for eq_value, group in groupby(vq_index[:input_length], key=self.eq_key)]
        ).to(input_feature.device)
        # ex) lengths = [1, 1, 3, 1] means there are 4 unique items, 
        # and items from 3rd to 5th are equal (3 repeated items).
        for len_ in length:
            len_ = len_.item()
            cluster_len[len_] = cluster_len.get(len_, 0) + 1

        weightList = 1 / ( length.size(0) * length )
        # ex) weightList = [0.25, 0.25, 0.08, 0.25]
        # We reduce redundant items, utilizing unique items more
        avg_weights[:, :input_length] = torch.repeat_interleave(weightList, length)[:,None]
        # ex) avg_weights[0, 0] = [0.25, 0.25, 0.08, 0.08, 0.08, 0.25, 0, 0, 0, 0]
        # Match averaging weights to the input items
    
        outputs = (input_feature * avg_weights).sum(1) # Length dimension
        outputs = outputs[-1, :] # Squeeze dimension
        return outputs, token_dict, cluster_len, input_length

    def eq_key(self, x):
        # You can define your own key, for different matching.
        # Refer to https://docs.python.org/3/library/itertools.html#itertools.groupby
        if self._eq_key=='exact':
            return tuple(x)
        elif self._eq_key=='former':
            return x[0]
        elif self._eq_key=='later':
            return x[-1]
        elif self._eq_key=='or':
            return _VQ_OR(*x)
        elif self._eq_key=='margin':
            return _VQ_DIST(x, margin=self.margin)
        else:
            Exception


class VQAvgSuperPool(nn.Module):
    """
    Weighted Average by VQ indices.
    If any of index from two tuples are same, merge it(divide weight).
    """
    def __init__(self) -> None:
        super().__init__()
    
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
        outputs = torch.zeros((B, D), device=input_feature.device)
        for i, (vq_indices_, input_length) in enumerate(zip(vq_indices, input_lengths)):
            # import pdb;pdb.set_trace()
            weight_dict = {}

            groups = [list(group) for eq_value, group in groupby(vq_indices_[:input_length], key=lambda x: x[0])]
            merged_groups = list()
            while groups:
                surplus = list()
                group = groups.pop(0)
                new_group = list()
                while group:
                    vq = group.pop(0)
                    new_group.append(vq)
                    while groups:
                        _group = groups.pop(0)
                        extended = False
                        for _vq in _group:
                            if _vq[1] == vq[1]: 
                                group.extend(_group)
                                extended = True
                                break
                            elif _vq[1] > vq[1]: break
                        if not extended:
                            surplus.append(_group)
                merged_groups.append(new_group)
                groups = surplus
            
            # pdb.set_trace()
            for group in merged_groups:
                for key in group:
                    weight_dict[tuple(key)] = 1/len(group)/len(merged_groups)
            
            # pdb.set_trace()
            
            avg_weights[i,:,:input_length,:] = torch.tensor([weight_dict[tuple(index)] for index in vq_indices_[:input_length]], dtype=avg_weights.dtype, device=avg_weights.device)[None, :, None]
        
        
        outputs = (input_feature * avg_weights).sum(2) # Length dimension
        outputs = outputs[:, -1, :] # Squeeze dimension
        return outputs

    def extract_stats(self, input_feature:Tensor, input_length:Tensor, vq_index:Tensor):# ->List[Dict[str,int], List[int], List[int]]:
        """
        Input feature size should follow (n_layers, Length, Dimension)
        vq index size should follow (Length, 2)
        Return (vq_representation, token dictionary, cluster lengths, token length)
        """
        from itertools import groupby
        assert input_feature.dim() == 3, f"Input feature size is {input_feature.size()}, Should follows (Layer, Length, Dimension)"
        input_feature = input_feature[-1:] # Select last layer only
        N, L, D = input_feature.shape

        token_dict = dict()
        cluster_len = dict()

        vq_index = vq_index.tolist() # This enormously accelarates the groupby calculation.
        for vq_index_ in vq_index:
            # === Count index number ===
            token_dict[tuple(vq_index_)] = token_dict.get(tuple(vq_index_), 0) + 1

        avg_weights = torch.zeros((1, L, 1), device=input_feature.device)
        
        weight_dict = {}

        groups = [list(group) for eq_value, group in groupby(vq_index[:input_length], key=lambda x: x[0])]
        merged_groups = list()
        while groups:
            surplus = list()
            group = groups.pop(0)
            new_group = list()
            while group:
                vq = group.pop(0)
                new_group.append(vq)
                while groups:
                    _group = groups.pop(0)
                    extended = False
                    for _vq in _group:
                        if _vq[1] == vq[1]: 
                            group.extend(_group)
                            extended = True
                            break
                        elif _vq[1] > vq[1]: break
                    if not extended:
                        surplus.append(_group)
            merged_groups.append(new_group)
            groups = surplus
        
        # pdb.set_trace()
        for group in merged_groups:
            
            len_ = len(group)
            cluster_len[len_] = cluster_len.get(len_, 0) + 1

            for key in group:
                weight_dict[tuple(key)] = 1/len(group)/len(merged_groups)
        
        # pdb.set_trace()
        
        avg_weights[:,:input_length] = torch.tensor([weight_dict[tuple(index)] for index in vq_index[:input_length]], dtype=avg_weights.dtype, device=avg_weights.device)[None, :, None]
    
        # ex) lengths = [1, 1, 3, 1] means there are 4 unique items, 
        # and items from 3rd to 5th are equal (3 repeated items).
        
        outputs = (input_feature * avg_weights).sum(1) # Length dimension
        outputs = outputs[-1, :] # Squeeze dimension
        return outputs, token_dict, cluster_len, input_length


class VQSqueezedAvgPool(VQWeightedAvgPool):
    """
    Average by VQ indices.
    """
    def __init__(self) -> None:
        super().__init__()
    
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
        outputs = torch.zeros((B, D), device=input_feature.device)
        for i, (vq_indices_, input_length) in enumerate(zip(vq_indices, input_lengths)):
            vq_indices_ = vq_indices_[:input_length]
            sorted_indices = sorted(range(input_length), key=lambda x: tuple(vq_indices_[x]))
            restore_indices = sorted(range(input_length), key=lambda x: sorted_indices[x])
            vq_indices_ = sorted(vq_indices_)
            lengths = torch.tensor(
                [len(list(group)) for eq_value, group in groupby(vq_indices_, key=self.eq_key)]
            ).to(input_feature.device)
            # vq_indices_ = vq_indices_[sorted_indices]
            # _, counts = vq_indices_.unique_consecutive(dim=0, return_counts=True)
            # ex) vq_unq_counts = [1, 1, 3, 1] means there are 4 unique items, 
            # and items from 3rd to 5th are equal (3 repeated items).
            weightList = 1 / ( lengths.size(0) * lengths )
            # ex) weightList = [0.25, 0.25, 0.08, 0.25]
            # We reduce redundant items, utilizing unique items more
            sorted_avg_weights = torch.repeat_interleave(weightList, lengths)[:,None]
            restored_avg_weigths = sorted_avg_weights[restore_indices]
            avg_weights[i,:,:input_length,:] = restored_avg_weigths
            # ex) avg_weights[0, 0] = [0.25, 0.25, 0.08, 0.08, 0.08, 0.25, 0, 0, 0, 0]
            # Match averaging weights to the input items
        
        outputs = (input_feature * avg_weights).sum(2) # Length dimension
        outputs = outputs[:, -1, :] # Squeeze dimension
        return outputs


def vq_distance(x:List[int], y:List[int]):
    return int(x[0] != y[0]) + int(x[1] != y[1])


class _VQ_DIST(object):
    def __init__(self, values, margin=0):
        # VQ index (values[0],values[1])
        self.values = values
        self.margin = margin

    def __eq__(self, other):
        return vq_distance(self.values, other.values) <= self.margin


class _VQ_OR(object):
    def __init__(self, x, y):
        # VQ index (x,y)
        self.x = x
        self.y = y

    def __eq__(self, other):
        return self.x == other.x or self.y == other.y


class VQLocalProbAvgPool(nn.Module):
    """
    Average by 
    frequency of vq index in each training sample.
    """
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, input_feature:Tensor, input_lengths:Tensor, vq_indices:Tensor):
        """
        Input feature size should follow (Batch size, n_layers, Length, Dimension)
        vq index size should follow (Batch size, Length, 2)
        Return speech representation which follows (Batch size, Dimension)
        """
        from collections import Counter
        from itertools import groupby
        assert input_feature.dim() == 4, f"Input feature size is {input_feature.size()}, Should follows (Batch, Layer, Length, Dimension)"
        input_feature = input_feature[:,-1:] # Select last layer only
        B, N, L, D = input_feature.shape

        # This enormously accelarates the groupby calculation.
        vq_indices_x = vq_indices[:, :, 0].to(torch.int16).tolist() # (B, L)
        vq_indices_y = vq_indices[:, :, 1].to(torch.int16).tolist()
        
        vq_num_x = [Counter(vq_sample) for vq_sample in vq_indices_x] # (L, )
        vq_num_y = [Counter(vq_sample) for vq_sample in vq_indices_y]

        vq_freq_x = torch.tensor([list(map(lambda index: vq_num_x[i][index], vq_indices_x[i])) for i in range(B)]) # (B, L)
        vq_freq_y = torch.tensor([list(map(lambda index: vq_num_y[i][index], vq_indices_y[i])) for i in range(B)])

        vq_factor =  1 / (vq_freq_x + vq_freq_y)
        avg_weights = F.softmax(vq_factor.log(), dim=-1).to(input_feature.device)
        avg_weights.unsqueeze_(1)
        avg_weights.unsqueeze_(3)

        outputs = (input_feature * avg_weights).sum(2) # Length dimension
        outputs = outputs[:, -1, :] # Squeeze dimension
        return outputs    


class VQGlobalProbAvgPool(nn.Module):
    """
    Average by 
    frequency of vq indices in the whole training set.
    """
    def __init__(self, a=0.5, freq_path='models/freq.pt') -> None:
        super().__init__()
        freqs = torch.load(freq_path, map_location='cpu')
        self.global_count = freqs
        self.global_count_x = freqs.sum(dim=1)
        self.global_count_y = freqs.sum(dim=0)
    
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

        # =============
        # VQ Global Weight
        # =============
        vq_indices_x = vq_indices[:, :, 0].to(torch.int16).tolist() # (B, L)
        vq_indices_y = vq_indices[:, :, 1].to(torch.int16).tolist()
        
        vq_freq_x = torch.tensor([list(map(lambda index: self.global_count_x[index], vq_indices_x[i])) for i in range(B)]) # (B, L)
        vq_freq_y = torch.tensor([list(map(lambda index: self.global_count_y[index], vq_indices_y[i])) for i in range(B)])

        vq_factor =  1 / (vq_freq_x + vq_freq_y)
        avg_weights = F.softmax(vq_factor.log(), dim=-1).to(input_feature.device)
        avg_weights.unsqueeze_(1)
        avg_weights.unsqueeze_(3)

        outputs = (input_feature * avg_weights).sum(2) # Length dimension
        outputs = outputs[:, -1, :] # Squeeze dimension
        return outputs


class VQMixedProbAvgPool(nn.Module):
    """
    Average by 
    (1) frequency of vq indices in the whole training set.
    (2) frequency of vq index in each training sample.
    """
    def __init__(self, a=0.5, freq_path='models/freq.pt') -> None:
        super().__init__()
        freqs = torch.load(freq_path, map_location='cpu')
        self.global_count = freqs
        self.global_count_x = freqs.sum(dim=1)
        self.global_count_y = freqs.sum(dim=0)

        self.a = a
    
    def forward(self, input_feature:Tensor, input_lengths:Tensor, vq_indices:Tensor):
        """
        Input feature size should follow (Batch size, n_layers, Length, Dimension)
        vq index size should follow (Batch size, Length, 2)
        Return speech representation which follows (Batch size, Dimension)
        """
        assert input_feature.dim() == 4, f"Input feature size is {input_feature.size()}, Should follows (Batch, Layer, Length, Dimension)"
        input_feature = input_feature[:,-1:] # Select last layer only
        B, N, L, D = input_feature.shape

        vq_indices_x = vq_indices[:, :, 0].to(torch.int16).tolist() # (B, L)
        vq_indices_y = vq_indices[:, :, 1].to(torch.int16).tolist()
        
        # =============
        # VQ Local Weight
        # =============
        from collections import Counter
        vq_num_x = [Counter(vq_sample) for vq_sample in vq_indices_x] # (L, )
        vq_num_y = [Counter(vq_sample) for vq_sample in vq_indices_y]

        vq_freq_local_x = torch.tensor([list(map(lambda index: vq_num_x[i][index], vq_indices_x[i])) for i in range(B)]) # (B, L)
        vq_freq_local_y = torch.tensor([list(map(lambda index: vq_num_y[i][index], vq_indices_y[i])) for i in range(B)])
        local_weight = 1 / (vq_freq_local_x + vq_freq_local_y)
        local_weight = F.softmax(local_weight.log(), dim=-1).to(input_feature.device)
        # =============
        # VQ Global Weight
        # =============
        vq_freq_global_x = torch.tensor([list(map(lambda index: self.global_count_x[index], vq_indices_x[i])) for i in range(B)]) # (B, L)
        vq_freq_global_y = torch.tensor([list(map(lambda index: self.global_count_y[index], vq_indices_y[i])) for i in range(B)])
        global_weight = 1 / (vq_freq_global_x + vq_freq_global_y)
        global_weight = F.softmax(global_weight.log(), dim=-1).to(input_feature.device)
        # =============
        # Mix
        # =============
        avg_weights = F.softmax(global_weight * local_weight, dim=-1)
        avg_weights /= avg_weights.sum(dim=-1, keepdim=True)
        avg_weights.unsqueeze_(1)
        avg_weights.unsqueeze_(3)

        outputs = (input_feature * avg_weights).sum(2) # Length dimension
        outputs = outputs[:, -1, :] # Squeeze dimension
        return outputs    


class ProbWeightedAvgPool(nn.Module):
    """
    Average by frequency.
    """
    def __init__(self, a=0.5, freq_path='models/freq.pt') -> None:
        super().__init__()
        freqs = torch.load(freq_path, map_location='cpu')
        probs = freqs/freqs.sum()
        self.weight = a / (probs + a)
    
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
        avg_weights = list()
        # avg_weights = torch.zeros((B, L), device=input_feature.device)
        
        def get_prob_normalized_weight(index:Tuple):
            index = tuple(map(int, index))
            return self.weight[index[0], index[1]]
        
        for i, (vq_indices_, input_length) in enumerate(zip(vq_indices, input_lengths)):
            # import pdb; pdb.set_trace()
            avg_weights.append(list(map(get_prob_normalized_weight, vq_indices_[:input_length])) + [0 for _ in range(len(vq_indices_) - input_length)])
            # for j in range(input_length):
            #     avg_weights[i, j] = get_prob_normalized_weight(vq_indices_[j])
            # avg_weights[i] /= avg_weights[i].sum()
        avg_weights = torch.tensor(avg_weights, device=input_feature.device)
        avg_weights /= avg_weights.sum(dim=-1, keepdim=True)
        avg_weights.unsqueeze_(1)
        avg_weights.unsqueeze_(3)

        outputs = (input_feature * avg_weights).sum(2) # Length dimension
        outputs = outputs[:, -1, :] # Squeeze dimension
        return outputs    


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
        assert input_feature.dim() == 4, f"Input feature size is {input_feature.size()}, Should follows (Batch, Layer, Length, Dimension)"
        input_feature = input_feature[:,-1] if input_feature.dim() == 4 else input_feature

        B, L, _ = input_feature.shape # (Batch size, Length, Dimension)

        h = torch.tanh(self.sap_linear(input_feature)) # (Batch size, Length, Dimension)
        w = torch.matmul(h, self.attention).squeeze(dim=2) # (Batch size, Length)
        w = F.softmax(w, dim=1).view(B, L, 1) # 
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
        assert input_feature.dim() == 4, f"Input feature size is {input_feature.size()}, Should follows (Batch, Layer, Length, Dimension)"
        input_feature = input_feature[:,-1] if input_feature.dim() == 4 else input_feature

        B, L, _ = input_feature.shape # (Batch size, Length, Dimension)

        h = torch.tanh(self.sap_linear(input_feature)) # (Batch size, Length, Dimension)
        w = torch.matmul(h, self.attention).squeeze(dim=2) # (Batch size, Length)

        # If length = 2, mask = [[1, 1, 0, 0, 0, ...]]
        mask = torch.arange(L).expand(B, L).to(w.device) < input_lengths[:,None] # result : (Batch size, Length)
        w = w + (~mask) * (w.min() - 20)

        w = F.softmax(w, dim=1).view(B, L, 1) # 
        feature = torch.sum(input_feature * w, dim=1) 

        return feature
    

class VQOneHotAttentivePooling(nn.Module):
    """
    Average by VQ indices.
    """
    def __init__(self, shrink='exact', input_dim = 768) -> None:
        super().__init__()
        self._eq_key = shrink

        VQ_MAX_INDEX = 320
        
        assert input_dim%2 == 0, "input_dim must be even"

        self.linear_former = nn.Linear(VQ_MAX_INDEX, int(input_dim / 2)) # 320, D
        self.linear_later = nn.Linear(VQ_MAX_INDEX, int(input_dim / 2))

        # Q, K, V
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
    
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

        vq_indices = Tensor.long(vq_indices) # modify dtype float32 -> int64

        # one-hot encoding
        vq_indices_one_hot = torch.nn.functional.one_hot(vq_indices, num_classes = 320) # B, L, 2, 320
        vq_indices_one_hot = Tensor.float(vq_indices_one_hot) # modify dtype int64 -> float32

        vq_feats_former = torch.tanh(self.linear_former(vq_indices_one_hot[:,:,0,:])) # B, L, D/2
        vq_feats_later = torch.tanh(self.linear_later(vq_indices_one_hot[:,:,1,:])) # B, L, D/2

        # concatenate
        vq_feats = torch.cat([vq_feats_former, vq_feats_later], dim = -1) # B, L, D

        query = self.query(vq_feats) # B, L, D
        key = self.key(vq_feats) # B, L, D
        value = self.value(input_feature.squeeze()) # B, L, D

        # calculate scaled-dot attention
        score = torch.bmm(query, key.permute(0, 2, 1)) / torch.sqrt(torch.tensor(D)) # B, L, L
        attention_weights = torch.softmax(score, dim = -1)

        outputs = torch.bmm(attention_weights, value) # (B, L, L) x (B, L, D) -> (B, L, D)

        return outputs.sum(1)
    
    
    
class XVector(nn.Module):
    def __init__(self, input_dim = 40, num_classes=8):
        super(XVector, self).__init__()
        from models.tasks.asv.tdnn import TDNN

        self.tdnn1 = TDNN(input_dim=input_dim, output_dim=512, context_size=5, dilation=1,dropout_p=0.5)
        self.tdnn2 = TDNN(input_dim=512, output_dim=512, context_size=3, dilation=1,dropout_p=0.5)
        self.tdnn3 = TDNN(input_dim=512, output_dim=512, context_size=2, dilation=2,dropout_p=0.5)
        self.tdnn4 = TDNN(input_dim=512, output_dim=512, context_size=1, dilation=1,dropout_p=0.5)
        self.tdnn5 = TDNN(input_dim=512, output_dim=512, context_size=1, dilation=3,dropout_p=0.5)
        #### Frame levelPooling
        self.segment6 = nn.Linear(1024, 512)
        self.segment7 = nn.Linear(512, 512)
        self.output = nn.Linear(512, num_classes)

    def forward(self, input_feature:Tensor, input_lengths:Tensor, *args):
        assert input_feature.dim() == 4, f"Input feature size is {input_feature.size()}, Should follows (Batch, Layer, Length, Dimension)"
        
        import pdb; pdb.set_trace()
        input_feature = input_feature[:,-1] if input_feature.dim() == 4 else input_feature
        
        tdnn1_out = self.tdnn1(input_feature)
        tdnn2_out = self.tdnn2(tdnn1_out)
        tdnn3_out = self.tdnn3(tdnn2_out)
        tdnn4_out = self.tdnn4(tdnn3_out)
        tdnn5_out = self.tdnn5(tdnn4_out)

        ### Stat Pool
        mean = torch.mean(tdnn5_out,1)
        std = torch.std(tdnn5_out,1)
        stat_pooling = torch.cat((mean,std),1)
        segment6_out = self.segment6(stat_pooling)
        x_vec = self.segment7(segment6_out)

        return x_vec




class WhiteningBERT(nn.Module):
    def __init__(self, layer_ids:List[int]=None, whitening=True, mean_vector_path:str=None, whiten_factor_path:str=None, normalize:str=None):
        super().__init__()
        print("layers", layer_ids)
        print("whitening", whitening)
        self.pool = AvgPool()
        self.layer_comb = LayerCombination(layer_ids=layer_ids)
        self.whitening = Whitening(mean_vector_path=mean_vector_path, whiten_factor_path=whiten_factor_path, normalize=normalize) if whitening else nn.Identity()

    def forward(self, input_feature:Tensor, input_lengths:Tensor, *args, **kwargs):
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
    def __init__(self, mean_vector_path:str=None, whiten_factor_path:str=None, normalize:str=None):
        super().__init__()
        self.mean_vector = torch.load(mean_vector_path)
        self.whiten_factor = torch.load(whiten_factor_path)
        self.normalize = normalize
        if self.normalize is not None:
            print(f'Normalize by {self.normalize}')
        
    def forward(self, input_feature:Tensor):
        """
        Input feature size should follow (Batch size, Dimension)
        Return speech representation which follows (Batch size, Dimension)
        """
        if self.mean_vector is None:
            self.mean_vector = input_feature.mean(dim=0, keepdim=True) # Batch mean    
        else:
            self.mean_vector = self.mean_vector.to(input_feature.device)
        
        if self.whiten_factor is None:
            Cov = torch.mm((input_feature-self.mean_vector).T,(input_feature-self.mean_vector)) # Global mean
            # eigval, eigvec = torch.linalg.eig(Cov)
            # D = torch.diag(eigval ** -0.5)
            # whiten_feature = (input_feature - m) @ eigvec @ D
            U, S, V = torch.svd(Cov)
            D = torch.diag(1/torch.sqrt(S))
            self.whiten_factor = torch.mm(U,D)
        else:
            self.whiten_factor = self.whiten_factor.to(input_feature.device)
        
        whiten_feature = torch.mm(input_feature - self.mean_vector, self.whiten_factor)
        
        if self.normalize == 'L2':
            return whiten_feature / torch.sum(whiten_feature**2, dim=1, keepdim=True)**0.5
        else:
            return whiten_feature
    
    def stack_feats(self, input_feature:Tensor):
        feat_sum = input_feature.sum(dim=0)
        self.datalen += input_feature.size(0)

        if self.features is None:
            self.features = feat_sum
        else:
            self.features += feat_sum

    def init_stats(self):
        self.mean_vector = self.features / self.datalen
        self.features = None


class SoftDecayPooling(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer_id= -1
        self.pool = AvgPool()

    def forward(self, input_feature:Tensor, input_lengths:Tensor, *args, **kwargs):
        """
        Input feature size should follow (Batch size, n_layers, Length, Dimension)
        Return speech representation which follows (Batch size, Dimension)
        """

        # Follow implemetation
        input_feature = self.pool(input_feature, input_lengths) # B, N, D
        input_feature = input_feature[:, self.layer_id] # B, D

        u,s,v = torch.svd(input_feature)
        #=== Soft Decay ===
        maxS = torch.max(s,dim=0).values.unsqueeze(-1)
        eps = 1e-7
        alpha = -0.6
        newS = - torch.log(1 - alpha * (s + alpha)+eps) / alpha
        #=== Rescaling ===
        maxNewS = torch.max(newS,dim=0).values.unsqueeze(-1)
        rescale_number = maxNewS/maxS
        newS = newS/rescale_number
        #=== Transform ===
        rescale_s_dia = torch.diag_embed(newS,dim1=-2,dim2=-1)
        output = torch.matmul(torch.matmul(u,rescale_s_dia),v.transpose(1,0))
        return output


        # Followed paper not implementation
        # outputs = torch.zeros_like(input_feature)
        # import pdb;pdb.set_trace()

        # for i, single_feature in enumerate(input_feature):
        #     u,s,v = torch.svd(single_feature.T)
        #     #=== Soft Decay ===
        #     maxS = torch.max(s,dim=0).values.unsqueeze(-1)
        #     eps = 1e-7
        #     alpha = -0.6
        #     newS = - torch.log(1 - alpha * (s + alpha)+eps) / alpha
        #     #=== Rescaling ===
        #     maxNewS = torch.max(newS,dim=0).values.unsqueeze(-1)
        #     rescale_number = maxNewS/maxS
        #     newS = newS/rescale_number
        #     #=== Transform ===
        #     rescale_s_dia = torch.diag_embed(newS,dim1=-2,dim2=-1)
        #     new_input = torch.matmul(torch.matmul(u,rescale_s_dia),v.transpose(1,0))
        #     outputs[i] = new_input.T

        # outputs = self.pool(outputs)
        # return outputs


class TFSqueezePooling(nn.Module):
    def __init__(self, n_dim:int=786, n_temporal:int=6) -> None:
        super().__init__()
        assert n_dim % n_temporal == 0, "n_dim should be divisible by n_temporal"
        self.layer_id= -1
        self.n_temporal = n_temporal
        self.avg_len = n_dim % self.n_temporal

    def forward(self, input_feature:Tensor, input_lengths:Tensor, *args, **kwargs):
        """
        Input feature size should follow (Batch size, n_layers, Length, Dimension)
        Return speech representation which follows (Batch size, Dimension)
        """
        input_feature = input_feature[:, -1]
        time_squeezed_feature = torch.cat([input_feature[:,i*self.avg_len:(i+1)*self.avg_len].mean(1, keepdim=True) for i in range(self.n_temporal)], dim=1)
        dim_squeezed_feature = torch.cat([time_squeezed_feature[:,:,i*self.n_temporal:(i+1)*self.n_temporal].mean(-1, keepdim=True) for i in range(int(input_feature.size(-1)/self.n_temporal))], dim=-1)
        squeezed_feature = dim_squeezed_feature.flatten(start_dim=1, end_dim=2) # B, D
        return squeezed_feature


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
    elif head_type=='x_vec':
        print( kwargs)
        return XVector(input_dim = input_dim, num_classes = 1211)
    elif head_type=='sap_mask':
        return SelfAttentiveMaskingPooling(input_dim)
    elif head_type=='white':
        return WhiteningBERT(layer_ids=layer_ids, whitening=kwargs['whitening'], mean_vector_path=kwargs['mean_vector_path'], whiten_factor_path=kwargs['whiten_factor_path'])
    elif head_type=='vq':
        return VQWeightedAvgPool(shrink=kwargs['shrink'])
    elif head_type=='vq_squeeze':
        return VQSqueezedAvgPool()
    elif head_type=='vq_super':
        return VQAvgSuperPool()
    elif head_type=='vq_one_hot_ap':
        return VQOneHotAttentivePooling(input_dim = input_dim)
    elif head_type=='prob':
        return ProbWeightedAvgPool(a=kwargs.get('a'), freq_path=kwargs['freq_path'])
    elif head_type=='softdecay':
        return SoftDecayPooling()
    elif head_type=='vq_local':
        return VQLocalProbAvgPool()
    elif head_type=='vq_global':
        return VQGlobalProbAvgPool(a=kwargs.get('a'), freq_path=kwargs['freq_path'])
    elif head_type=='vq_mix':
        return VQMixedProbAvgPool(a=kwargs.get('a'), freq_path=kwargs['freq_path'])
    else:
        assert False, f"""HEAD TYPE "{head_type}" IS NOT IMPLEMENTED!"""