import os
import sys
import tqdm

import torch

from dataloader.utils import load_dataset
from dataloader.dataset import DATA_LIST
from models.feats_extractor import *


def main(data_name:str, root:str='data', new_root:str='data2', url:str=None, subset:str=None, tag:str=None):
    assert data_name in DATA_LIST, f"{data_name} IS NOT EXISTING DATASET!!"

    DEVICE = 'cuda'if torch.cuda.is_available() else 'cpu'
    dataset = load_dataset(root=root, data_name=data_name, url=url, subset=subset, ext='wav') # should return (an audio array, sampling rate)
    extractor = VQWav2VecExtractor().to(device=DEVICE)
    extractor.eval()
    for i in tqdm.tqdm(range(len(dataset))):
        new_path = dataset.generate_feature_path(i, new_root=new_root, tag=tag)
        waveform, sr = dataset[i]
        waveform = waveform.to(DEVICE)
        with torch.no_grad():
            features = extractor.extract(waveform, sr).detach().cpu()
        # import pdb; pdb.set_trace()
        torch.save(features, new_path)
        

if __name__=='__main__':
    main(*sys.argv[1:])