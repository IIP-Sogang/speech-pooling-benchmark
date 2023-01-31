import glob
from typing import List, Optional, Tuple, Callable

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from .dataset import SpeechCommandDataset


def generate_librispeech_metadata(data_dir):
    transcript_paths = glob.glob(f"{data_dir}/*/*/*.trans.txt")
    audio_paths = glob.glob(f"{data_dir}/*/*/*.flac")

    transcript_dict = dict()

    for trans_path in transcript_paths:
        with open(trans_path) as f:
            for line in f:
                filename, transcript = f.strip().split()
                transcript_dict[filename] = transcript
 
    ### NOTE (JK) 구현중..

def load_dataset(data_name:str='speechcommands', get_collate_fn:bool=False, **kwargs)->Tuple[torch.utils.data.Dataset, Optional[Callable]]:
    if data_name == 'speechcommands':
        for key in ['root', 'subset']:
            assert key in kwargs, f"Pass '{key}' through the config yaml file!!"
        dataset = SpeechCommandDataset(**kwargs)
        if get_collate_fn:
            return dataset, pad_collate
        else:
            return dataset
    else:
        assert False, f"DATA '{data_name}' IS NOT IMPLEMENTD!"

def pad_collate(batch:List[Tuple[Tensor, int]]):
    batch_size = len(batch)
    max_array_length = 0

    for array, _ in batch:
        if len(array)>max_array_length: max_array_length = len(array)

    data = torch.zeros((batch_size, max_array_length, batch[0][0].size(-1)))
    labels = torch.zeros((batch_size, ), dtype=torch.long)
    for i, (array, label) in enumerate(batch):
        data[i, :len(array)] = array
        labels[i] = label

    return data, labels