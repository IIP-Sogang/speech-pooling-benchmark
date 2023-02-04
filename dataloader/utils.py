import glob
from typing import List, Optional, Tuple, Callable

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from .dataset import SpeechCommandDataset, VoxCelebDataset



def load_dataset(data_name:str='speechcommands', get_collate_fn:bool=False, **kwargs)->Tuple[torch.utils.data.Dataset, Optional[Callable]]:
    if data_name == 'speechcommands':
        for key in ['root', 'subset']:
            assert key in kwargs, f"Pass '{key}' through the config yaml file!!"
        dataset = SpeechCommandDataset(**kwargs)
        if get_collate_fn:
            return dataset, pad_collate
        else:
            return dataset
    elif data_name == 'voxceleb':
        for key in ['root', 'subset']:
            assert key in kwargs, f"Pass '{key}' through the config yaml file!!"
        dataset = VoxCelebDataset(**kwargs)
        if get_collate_fn:
            return dataset, pad_collate
        else:
            return dataset
    else:
        assert False, f"DATA '{data_name}' IS NOT IMPLEMENTD!"

def pad_collate(batch:List[Tuple[Tensor, int]]):
    batch_size = len(batch)
    batch_sample = batch[0][0]
    batch_dim = len(batch_sample.shape)
    
    max_array_length = 0

    search_dim = 0 if batch_dim == 2 else 1
    for array, _ in batch:
        if array.size(search_dim)>max_array_length: max_array_length = array.size(search_dim)

    data = torch.zeros((batch_size, max_array_length, batch_sample.size(-1))) if batch_dim == 2 \
           else torch.zeros((batch_size, batch_sample.size(0), max_array_length, batch_sample.size(-1)))
    labels = torch.zeros((batch_size, ), dtype=torch.long)
    
    for i, (array, label) in enumerate(batch):
        if batch_dim == 2:
            data[i, :len(array)] = array
        else:
            data[i, :, :array.size(1)] = array
        labels[i] = label

    return data, labels