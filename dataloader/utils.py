import os
from typing import List, Optional, Tuple, Callable

import torch
import torch.nn as nn
import torchaudio
from torch import Tensor
from torch.utils.data import DataLoader

from .dataset import SpeechCommandDataset, VoxCelebDataset, IEMOCAPDataset, VoxCelebVerificationDataset, FluentSpeechCommandsDataset


def load_dataset(data_name:str='speechcommands', get_collate_fn:bool=False, **kwargs)->Tuple[torch.utils.data.Dataset, Optional[Callable]]:
    method = kwargs['method']
    
    if data_name == 'speechcommands':
        for key in ['root', 'subset']:
            assert key in kwargs, f"Pass '{key}' through the config yaml file!!"
        dataset = SpeechCommandDataset(**kwargs)

        if get_collate_fn & (method != 'conv_feature'): # use transformer feature
            return dataset, pad_collate
        
        elif get_collate_fn & (method == 'conv_feature'): # use vector quantization
            return dataset, pad_collate_vq
        
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

        if get_collate_fn & (method != 'conv_feature'): # use transformer feature
            return dataset, pad_collate
        
        elif get_collate_fn & (method == 'conv_feature'): # use vector quantization
            return dataset, pad_collate_vq
        
        else:
            return dataset

    elif data_name == 'fluent':
        for key in ['root', 'subset']:
            assert key in kwargs, f"Pass '{key}' through the config yaml file!!"
        dataset = FluentSpeechCommandsDataset(**kwargs)

        if get_collate_fn & (method != 'conv_feature'): # use transformer feature
            return dataset, pad_collate
        
        elif get_collate_fn & (method == 'conv_feature'): # use vector quantization
            return dataset, pad_collate_vq
        
        else:
            return dataset
    else:
        assert False, f"DATA '{data_name}' IS NOT IMPLEMENTD!"


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

def pad_collate_vq(batch:List[Tuple[Tensor, int]]):
    # batch : [batch, [transformer_features, conv_features, label]]

    batch_size = len(batch)
    batch_transformer_sample = batch[0][0]
    batch_dim = len(batch_transformer_sample.shape)


    search_dim = 0 if batch_dim == 2 else 1 # 1

    transformer_data_lengths = torch.zeros((batch_size,), dtype=torch.long)
    conv_data_lengths = torch.zeros((batch_size,), dtype=torch.long)
    for i, (transformer_array, conv_array, _) in enumerate(batch):
        transformer_data_lengths[i] = transformer_array.size(search_dim) # transformer_array : [2, time frame, channel], in that case search dimmension is 1
        conv_data_lengths[i] = conv_array.size(search_dim -1) # conv_array : [time frame, 2]
    transformer_max_array_length = transformer_data_lengths.max()
    conv_max_array_length = conv_data_lengths.max()

    assert transformer_max_array_length == conv_max_array_length, "warning!! time frame of transformer is not equal to conv feature's frame"

    transformer_data = torch.zeros((batch_size, transformer_max_array_length, batch_transformer_sample.size(-1))) if batch_dim == 2 \
           else torch.zeros((batch_size, batch_transformer_sample.size(0), transformer_max_array_length, batch_transformer_sample.size(-1)))
    
    # conv_data
    if batch_dim == 2:
        raise ValueError("check batch dimmension of conv_data")
    elif batch_dim == 3:    
        conv_data = torch.zeros((batch_size, transformer_max_array_length, 2))

    labels = torch.zeros((batch_size, ), dtype=torch.long)
    
    for i, (tr_array, cv_array, label) in enumerate(batch):
        if batch_dim == 2:
            transformer_data[i, :len(tr_array)] = tr_array
            raise ValueError("check batch dimmension of conv_data")
        else:
            transformer_data[i, :, :tr_array.size(1), :] = tr_array
            conv_data[i, :cv_array.size(0), :] = cv_array
            assert tr_array.size(1) == cv_array.size(0), "check time frame size (@ collator)"
        labels[i] = label

    return transformer_data, conv_data, transformer_max_array_length, labels