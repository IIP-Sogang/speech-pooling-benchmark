import glob

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import SpeechCommandDataset, DATA_LIST


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


def load_dataloader(data_name:str='speechcommands', data_dir:str='data', subset='training'):
    if data_name == 'speechcommands':
        dataset= SpeechCommandDataset(subset=subset)

    return torch.utils.data.DataLoader(dataset)

def load_dataset(data_name:str='speechcommands', data_dir:str='data', subset='training'):
    if data_name == 'speechcommands':
        return SpeechCommandDataset(subset=subset)