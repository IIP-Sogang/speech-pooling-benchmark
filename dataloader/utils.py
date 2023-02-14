import os
from typing import List, Optional, Tuple, Callable

import torch
import torch.nn as nn
import torchaudio
from torch import Tensor
from torch.utils.data import DataLoader

from .dataset import _FluentSpeechCommandsDataset, SpeechCommandDataset, VoxCelebDataset, IEMOCAPDataset, VoxCelebVerificationDataset, FluentSpeechCommandsDataset


def load_dataset(data_name:str='speechcommands', get_collate_fn:bool=False, **kwargs)->Tuple[torch.utils.data.Dataset, Optional[Callable]]:    
    """
    get_collate_fn=False option is used at *_extract_feats.py
    for extracting a single data
    """
    if data_name == 'speechcommands':
        for key in ['root', 'subset']:
            assert key in kwargs, f"Pass '{key}' through the config yaml file!!"
        dataset = SpeechCommandDataset(**kwargs)

        if dataset.return_vq:
            collate_fn = pad_collate_vq
        else:
            collate_fn = pad_collate
        
    elif data_name == 'voxceleb':
        for key in ['root', 'subset']:
            assert key in kwargs, f"Pass '{key}' through the config yaml file!!"
        
        # Select Dataset & collate function
        if kwargs['subset'] == 'training':
            dataset = VoxCelebDataset(**kwargs)
            if dataset.return_vq:
                collate_fn = pad_collate_vq
            else:
                collate_fn = pad_collate
        else:
            dataset = VoxCelebVerificationDataset(**kwargs)
            if dataset.return_vq:
                return pad_double_collate_vq
            else:
                collate_fn = pad_double_collate

    elif data_name == 'iemocap':
        for key in ['root']:
            assert key in kwargs, f"Pass '{key}' through the config yaml file!!"
        dataset = IEMOCAPDataset(**kwargs)

        if dataset.return_vq:
            collate_fn = pad_collate_vq
        else:
            collate_fn = pad_collate
  
    elif data_name == 'fluent':
        for key in ['root', 'subset']:
            assert key in kwargs, f"Pass '{key}' through the config yaml file!!"
        dataset = FluentSpeechCommandsDataset(**kwargs)

        if dataset.return_vq:
            collate_fn = pad_collate_vq
        else:
            collate_fn = pad_collate

    elif data_name == '_fluent':
        print("Warning: Slot Mode is Deprecated")
        for key in ['root', 'subset']:
            assert key in kwargs, f"Pass '{key}' through the config yaml file!!"
        dataset = _FluentSpeechCommandsDataset(**kwargs)
        if dataset.return_vq:
            raise NotImplementedError
        else:
            collate_fn = pad_collate_slot

    else:
        assert False, f"DATA '{data_name}' IS NOT IMPLEMENTD!"

    if get_collate_fn:
        return dataset, collate_fn
    else: 
        return dataset


def pad_collate(batch:List[Tuple[Tensor, int]]):
    batch_size = len(batch)
    batch_sample = batch[0][0] # [2, 127, 768]
    batch_dim = len(batch_sample.shape) # 3 (@ using transformer feature)
    
    max_array_length = 0

    search_dim = 0 if batch_dim == 2 else 1 # search_dim = 1
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

def pad_collate_slot(batch:List[Tuple[Tensor, int]]):
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
    labels = torch.zeros((batch_size, 3), dtype=torch.long)
    
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

def pad_collate_vq(batch:List[Tuple[Tensor, int, Tensor]]):
    batch_size = len(batch)
    batch_sample = batch[0][0] # [2, 127, 768]
    batch_dim = len(batch_sample.shape) # 3 (@ using transformer feature)
    
    max_array_length = 0

    search_dim = 0 if batch_dim == 2 else 1 # search_dim = 1
    data_lengths = torch.zeros((batch_size,), dtype=torch.long)
    for i, (array, _, _) in enumerate(batch):
        data_lengths[i] = array.size(search_dim)
    max_array_length = data_lengths.max()

    data = torch.zeros((batch_size, max_array_length, batch_sample.size(-1))) if batch_dim == 2 \
           else torch.zeros((batch_size, batch_sample.size(0), max_array_length, batch_sample.size(-1)))
    vq_index = torch.zeros((batch_size, max_array_length, 2))
    labels = torch.zeros((batch_size, ), dtype=torch.long)
    
    for i, (array, label, vq_array) in enumerate(batch):
        if batch_dim == 2:
            data[i, :len(array)] = array
        else:
            data[i, :, :array.size(1)] = array
        vq_index[i, :vq_array.size(0)] = vq_array
        labels[i] = label

    return data, data_lengths, vq_index, labels


def pad_double_collate_vq(batch:List[Tuple[Tensor, Tensor, int, Tensor, Tensor]]):
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
    vq_index = torch.zeros((2, batch_size, max_array_length, 2))
    labels = torch.zeros((batch_size, ), dtype=torch.long)
    
    for i, (array1, array2, label, vq_array1, vq_array2) in enumerate(batch):
        if batch_dim == 2:
            data[0, i, :len(array1)] = array1
            data[1, i, :len(array2)] = array2
        else:
            data[0, i, :, :array1.size(1)] = array1
            data[1, i, :, :array2.size(1)] = array2
        vq_index[0, i, :vq_array1.size(0)] = vq_array1
        vq_index[1, i, :vq_array2.size(0)] = vq_array2
        labels[i] = label

    return data, data_lengths, vq_index, labels
