import os
from typing import List, Optional, Tuple, Callable

import torch
import torch.nn as nn
import torchaudio
from torch import Tensor
from torch.utils.data import DataLoader

from .dataset import SpeechCommandDataset, VoxCelebDataset, IEMOCAPDataset, VoxCelebVerificationDataset, FluentSpeechCommandsDataset


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
        
        # Select Dataset & collate function
        if kwargs['subset'] == 'training':
            dataset = VoxCelebDataset(**kwargs)
            collate_fn = pad_collate
        else:
            dataset = VoxCelebVerificationDataset(**kwargs)
            collate_fn = pad_double_collate

        if get_collate_fn:
            return dataset, collate_fn
        else:
            return dataset

    elif data_name == 'iemocap':
        for key in ['root']:
            assert key in kwargs, f"Pass '{key}' through the config yaml file!!"
        dataset = IEMOCAPDataset(**kwargs)
        if get_collate_fn:
            return dataset, pad_collate
        else:
            return dataset

    elif data_name == 'fluent':
        for key in ['root', 'subset']:
            assert key in kwargs, f"Pass '{key}' through the config yaml file!!"
        dataset = FluentSpeechCommandsDataset(**kwargs)
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
    data_lengths = torch.zeros((batch_size,), dtype=torch.long)
    for i, (array, _) in enumerate(batch):
        data_lengths[i] = array.size(search_dim)
    max_array_length = data_lengths.max()

    data = torch.zeros((batch_size, max_array_length, batch_sample.size(-1))) if batch_dim == 2 \
           else torch.zeros((batch_size, batch_sample.size(0), max_array_length, batch_sample.size(-1)))
    labels = torch.zeros((batch_size, ), dtype=torch.long)
    
    for i, (array, label) in enumerate(batch):
        if batch_dim == 2:
            data[i, :len(array)] = array
        else:
            data[i, :, :array.size(1)] = array
        labels[i] = label

    return data, data_lengths, labels


def pad_double_collate(batch:List[Tuple[Tensor, Tensor, int]]):
    """
    Return
    data : Tensor (2, batch size, length) or (2, batch size, dimension, length)
    data_lengths : Tensor (2, batch size) or (2, batch size)
    labels : Tensor (batch size, )
    """
    batch_size = len(batch)
    batch_sample = batch[0][0]
    batch_dim = len(batch_sample.shape)
    
    max_array_length = 0

    search_dim = 0 if batch_dim == 2 else 1
    data_lengths = torch.zeros((2, batch_size,), dtype=torch.long)
    for i, (array1, array2, _) in enumerate(batch):
        for j, array in enumerate([array1, array2]):
            data_lengths[j][i] = array.size(search_dim)
    max_array_length = data_lengths.max()

    data = torch.zeros((2, batch_size, max_array_length, batch_sample.size(-1))) if batch_dim == 2 \
           else torch.zeros((2, batch_size, batch_sample.size(0), max_array_length, batch_sample.size(-1)))
    labels = torch.zeros((batch_size, ), dtype=torch.long)
    
    for i, (array1, array2, label) in enumerate(batch):
        if batch_dim == 2:
            data[0, i, :len(array1)] = array1
            data[1, i, :len(array2)] = array2
        else:
            data[0, i, :, :array1.size(1)] = array1
            data[1, i, :, :array2.size(1)] = array2
        labels[i] = label

    return data, data_lengths, labels