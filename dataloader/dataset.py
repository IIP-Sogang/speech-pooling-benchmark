import os
import glob
from pathlib import Path
from typing import Tuple, Union

import torch
import torch.nn as nn
import torchaudio
from torch import Tensor

HASH_DIVIDER = "_nohash_"
EXCEPT_FOLDER = "_background_noise_"
DATA_LIST = [
    "speechcommands",
    "librispeech",
]

class LibreSpeechDataset(nn.Module):
    def __init__(self, meta_path:str):
        with open(meta_path) as f: f.readlines()


class SpeechCommandDataset(torchaudio.datasets.SPEECHCOMMANDS):
    CLASS_LIST = [
        'backward', 'bed', 'bird', 'cat', 'dog', 'down', 
        'eight', 'five', 'follow', 'forward', 'four', 'go', 
        'happy', 'house', 'learn', 'left', 'marvin', 'nine', 
        'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 
        'six', 'stop', 'three', 'tree', 'two', 'up', 
        'visual', 'wow', 'yes', 'zero']
    CLASS_DICT = {class_:i for i, class_ in enumerate(CLASS_LIST)}
    CLASS_DICT_INV = {value:key for key, value in CLASS_DICT.items()}

    def __init__(self, root:str='data', folder_in_archive='SpeechCommands', url='speech_commands_v0.02', subset:str='training', ext:str='wav', download=False):
        super().__init__(subset=subset, root=root, folder_in_archive=folder_in_archive, url=url, download=download)
        assert subset in ['training','validation','testing']
        assert os.path.exists(root+'/'+folder_in_archive)

        self.folder_in_archive = folder_in_archive
        self.ext = ext

        if subset == "validation": pass
        elif subset == "testing": pass
        elif subset == "training":
            excludes = set(_load_list(self._path, "validation_list.txt", "testing_list.txt"))
            walker = sorted(str(p) for p in Path(self._path).glob(f"*/*.{self.ext}"))
            self._walker = [
                w
                for w in walker
                if HASH_DIVIDER in w and EXCEPT_FOLDER not in w and os.path.normpath(w) not in excludes
            ]
        else:
            walker = sorted(str(p) for p in Path(self._path).glob(f"*/*.{self.ext}"))
            self._walker = [w for w in walker if HASH_DIVIDER in w and EXCEPT_FOLDER not in w]
        
    def __getitem__(self, n: int) -> Union[Tuple[Tensor, int, str, str, int], Tuple[Tensor, int]]:
        if self.ext == 'pt':
            pt_path = self.get_metadata(n)
            label:str = os.path.basename(os.path.dirname(pt_path))
            label:int = self.label2index(label)
            return (torch.load(pt_path, map_location='cpu'), label)
        else:
            return super().__getitem__(n)

    def get_metadata(self, index):
        return self._walker[index]

    def label2index(self, label):
        return self.CLASS_DICT[label]

    def index2label(self, index):
        return self.CLASS_DICT_INV[index]

    def generate_feature_path(self, index):
        old_path = self.get_metadata(index)
        new_path = old_path.replace(self.folder_in_archive, self.folder_in_archive+"_feat").replace('.wav','.pt')
        
        if not os.path.exists(os.path.dirname(new_path)):
            os.makedirs(os.path.dirname(new_path))

        return new_path



def _load_list(root, *filenames):
    output = []
    for filename in filenames:
        filepath = os.path.join(root, filename)
        with open(filepath) as fileobj:
            output += [os.path.normpath(os.path.join(root, line.strip())) for line in fileobj]
    return output