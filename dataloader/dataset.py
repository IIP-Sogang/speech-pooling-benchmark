import os
import re
import glob
from pathlib import Path
from typing import Tuple, Union, Optional

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
    "iemocap"
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
    VRF_TEST_SPEAKER_ID = ["id10270", "id10272", "id10274", "id10276", "id10278", "id10280", "id10282", "id10284", "id10286", "id10288", 
                           "id10290", "id10292", "id10294", "id10296", "id10298", "id10300", "id10302", "id10304", "id10306", "id10308", 
                           "id10271", "id10273", "id10275", "id10277", "id10279", "id10281", "id10283", "id10285", "id10287", "id10289", 
                           "id10291", "id10293", "id10295", "id10297", "id10299", "id10301", "id10303", "id10305", "id10307", "id10309"]

    def __init__(self, root:str='data', subset:str='training', url:str='vrfy_split.txt', ext:str='pt', download=False, **kwargs):
        assert subset in ['training','validation','testing']
        assert os.path.exists(root)
        subset = map_subset_voxceleb(subset)
        super().__init__(root=root, subset=subset, meta_url=url, download=download)
        self._ext_audio = '.'+ext
        self.root = root
        self.id2class = self._map_spk_id()

    def generate_feature_path(self, index, new_root:str='data/VoxCeleb1', tag:str='_feat'):
        old_path, _, _, _ = self.get_metadata(index)
        # new_path = old_path.replace(self.root, new_root+tag).replace('.wav','.pt')
        new_path = (new_root+tag+'/'+old_path).replace('.wav','.pt')
        
        if not os.path.exists(os.path.dirname(new_path)):
            os.makedirs(os.path.dirname(new_path))

        return new_path

    def __getitem__(self, n: int) -> Tuple[Tensor, int]:
        if self._ext_audio == '.wav':
            return super().__getitem__(n)[:2]
        else:
            metadata = self.get_metadata(n)
            pt = torch.load(os.path.join(self._path, metadata[0]))
            spk_label = self.id2class[metadata[2]]
            return (pt, spk_label) # (pt, spk_id)

    def _map_spk_id(self)->int:
        import os
        spks = list()
        with os.scandir(self.root) as it:
            for entry in it:
                if entry.is_dir() and entry.name not in self.VRF_TEST_SPEAKER_ID:
                    spks.append(int(entry.name[3:]))
        print(f"TOTAL {len(spks)} SPEAKERS ARE FOUND")
        return {spk_id:i for i, spk_id in enumerate(spks)}

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



# Emotion Recognition
def _get_wavs_paths(data_dir):
    wav_dir = data_dir / "sentences" / "wav"
    wav_paths = sorted(str(p) for p in wav_dir.glob("*/*.wav"))
    relative_paths = []
    for wav_path in wav_paths:
        start = wav_path.find("Session")
        wav_path = wav_path[start:]
        relative_paths.append(wav_path)
    return relative_paths


class IEMOCAPDataset(torchaudio.datasets.IEMOCAP):
    CLASS_LIST = ["neu", "hap", "ang", "sad"]
    CLASS_DICT = {class_:i for i, class_ in enumerate(CLASS_LIST)}
    CLASS_DICT_INV = {value:key for key, value in CLASS_DICT.items()}

    def __init__(
        self,
        root: Union[str, Path],
        sessions: Tuple[str] = (1, 2, 3, 4, 5),
        utterance_type: Optional[str] = None,
        ext:str='wav',
        feature_path_tag:str='_feat_1_12',
        final_classes: Tuple[str] = ("neu", "hap", "ang", "sad"),
        **kwargs,
    ):
        root = Path(root)
        self._path = root / "IEMOCAP"
        self.ext = ext
        self.feature_path_tag = feature_path_tag

        if not os.path.isdir(self._path):
            raise RuntimeError("Dataset not found.")

        if utterance_type not in ["scripted", "improvised", None]:
            raise ValueError("utterance_type must be one of ['scripted', 'improvised', or None]")

        all_data = []
        self.data = []
        self.mapping = {}

        for session in sessions:
            session_name = f"Session{session}"
            print(session_name) ###############
            session_dir = self._path / session_name

            # get wav paths
            wav_paths = _get_wavs_paths(session_dir)
            for wav_path in wav_paths:
                wav_stem = str(Path(wav_path).stem)
                all_data.append(wav_stem)

            # add labels
            label_dir = session_dir / "dialog" / "EmoEvaluation"
            query = "*.txt"
            if utterance_type == "scripted":
                query = "*script*.txt"
            elif utterance_type == "improvised":
                query = "*impro*.txt"
            label_paths = label_dir.glob(query)

            for label_path in label_paths:
                # ⚡ remove redundant files
                if '._' in str(label_path):
                    continue 

                with open(label_path, "r") as f:
                    for line in f:
                        if not line.startswith("["):
                            continue
                        line = re.split("[\t\n]", line)
                        wav_stem = line[1] # 'Ses01F_impro01_F000
                        label = line[2] # 'neu'
                        if wav_stem not in all_data: 
                            continue
                        if label not in final_classes: # ["neu", "hap", "ang", "sad", "exc", "fru"]
                            continue
                        self.mapping[wav_stem] = {}
                        self.mapping[wav_stem]["label"] = label

            for wav_path in wav_paths:
                wav_stem = str(Path(wav_path).stem)
                if wav_stem in self.mapping:
                    self.data.append(wav_stem)
                    self.mapping[wav_stem]["path"] = wav_path

    def __getitem__(self, n: int) -> Tuple[Tensor, int]:
        if self.ext == 'pt':
            new_root = str(self._path)
            pt_path = self.generate_feature_path(n, new_root = new_root, tag = self.feature_path_tag)
            wav_path, sr, wav_stem, label, speaker = self.get_metadata(n)
            emo_label = self.label2index(label)

            return (torch.load(pt_path, map_location='cpu'), emo_label)
        else:
            return super().__getitem__(n)[:2] #Tuple[Tensor, int, str, str, int]
        

    def generate_feature_path(self, index, new_root:str='/home/nas4/DB/IEMOCAP/IEMOCAP', tag:str='_feat'):
        wav_path, _, _, _, _ = self.get_metadata(index)
        old_path = str(self._path / wav_path)
        new_path = old_path.replace(str(self._path), new_root+tag).replace('.wav','.pt')
        
        if not os.path.exists(os.path.dirname(new_path)):
            os.makedirs(os.path.dirname(new_path))

        return new_path
    
    def label2index(self, label):
        return self.CLASS_DICT[label]

    def index2label(self, index):
        return self.CLASS_DICT_INV[index]
        
        


