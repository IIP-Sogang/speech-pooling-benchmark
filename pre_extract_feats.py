import os
import sys
import tqdm

import torch

from dataloader.utils import load_dataset, DATA_LIST
from models.feats_extractor import *


def main(data_name:str, subset:str=None):
    # device = 'cuda'if torch.cuda.is_available() else 'cpu'
    # new_data_dir = 'SpeechCommands_feat'
    assert data_name in DATA_LIST, f"{data_name} IS NOT EXISTING DATASET!!"

    dataset = load_dataset(data_name=data_name, subset=subset) # should return (an audio array, sampling rate)

    extractor = Wav2VecExtractor()
    extractor.eval()
    for i in tqdm.tqdm(range(len(dataset))):
        with torch.no_grad():
            features = extractor.extract(dataset[i])
        
        new_path = dataset.generate_feature_path(i)
        torch.save(features[-1][0], new_path) # Save last layer features


if __name__=='__main__':
    main(*sys.argv[1:])