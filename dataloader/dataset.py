import os
import glob
from pathlib import Path
from typing import Tuple, Union

import torch
import torch.nn as nn
import torchaudio
from torch import Tensor

from dataloader.voxceleb1 import VoxCeleb1Identification, VoxCeleb1Verification

HASH_DIVIDER = "_nohash_"
EXCEPT_FOLDER = "_background_noise_"
DATA_LIST = [
    "speechcommands",
    "voxceleb",
]


# Keyword Spotting
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
        
    def __getitem__(self, n: int) -> Tuple[Tensor, int]:
        if self.ext == 'pt':
            pt_path = self.get_metadata(n)
            label:str = os.path.basename(os.path.dirname(pt_path))
            label:int = self.label2index(label)
            return (torch.load(pt_path, map_location='cpu'), label)
        else:
            return super().__getitem__(n)[:2] #Tuple[Tensor, int, str, str, int]

    def get_metadata(self, index):
        return self._walker[index]

    def label2index(self, label):
        return self.CLASS_DICT[label]

    def index2label(self, index):
        return self.CLASS_DICT_INV[index]

    def generate_feature_path(self, index, tag:str='_feat'):
        old_path = self.get_metadata(index)
        new_path = old_path.replace(self.folder_in_archive, self.folder_in_archive + tag).replace('.wav','.pt')
        
        if not os.path.exists(os.path.dirname(new_path)):
            os.makedirs(os.path.dirname(new_path))

        return new_path

# Speaker Verification
def map_subset_voxceleb(subset:str):
    if subset=='training':
        return 'train'
    elif subset=='validation':
        return 'dev'
    elif subset=='testing':
        return 'test'
    else:
        raise Exception


class VoxCelebDataset(VoxCeleb1Identification):
    def __init__(self, root:str='data', subset:str='training', url:str='iden_split.txt', ext:str='wav', download=False):
        assert subset in ['training','validation','testing']
        assert os.path.exists(root)
        subset = map_subset_voxceleb(subset)
        super().__init__(root=root, subset=subset, meta_url=url, download=download)
        self._ext_audio = '.'+ext
        self.root = root

    def generate_feature_path(self, index, new_root:str='data/VoxCeleb1', tag:str='_feat'):
        old_path, _, _, _ = self.get_metadata(index)
        # new_path = old_path.replace(self.root, new_root+tag).replace('.wav','.pt')
        new_path = (new_root+tag+'/'+old_path).replace('.wav','.pt')
        
        if not os.path.exists(os.path.dirname(new_path)):
            os.makedirs(os.path.dirname(new_path))

        return new_path

    def __getitem__(self, n: int) -> Tuple[Tensor, int]:
        return super().__getitem__(n)[:2]


class VoxCelebVerificationDataset(VoxCeleb1Verification):
    def __init__(self, root:str='data', subset:str='training', ext:str='wav', download=False):
        assert subset in ['training','validation','testing']
        assert os.path.exists(root)
        
        super().__init__(root=root, meta_url='', download=download)
        self._ext_audio = '.'+ext
        self.root = root
        
    def __getitem__(self, n: int) -> Tuple[Tensor, Tensor, int, int, str, str]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded.

        Returns:
            Tuple of the following items;

            Tensor:
                Waveform of speaker 1
            Tensor:
                Waveform of speaker 2
            int:
                Sample rate
            int:
                Label
            str:
                File ID of speaker 1
            str:
                File ID of speaker 2
        """
        if self._ext_audio == '.pt':
            metadata = self.get_metadata(n)
            waveform_spk1 = torch.load(self._path, metadata[0], metadata[2])
            waveform_spk2 = torch.load(self._path, metadata[1], metadata[2])
            return (waveform_spk1, waveform_spk2) + metadata[2:]
        elif self._ext_audio == '.wav':
            return self.__getitem__(n)

    def generate_feature_path(self, index, new_root:str='data/VoxCeleb1', tag:str='_feat'):
        old_path = self.get_metadata(index)
        new_path = old_path.replace(self.root, new_root+tag).replace('.wav','.pt')
        
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