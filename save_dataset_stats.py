import os
import tqdm
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader

from dataloader.utils import load_dataset

# === ARGUMENTS ===

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--upstream', type=str, default='wav2vec2_large')
parser.add_argument('--dim', type=int, default=1024)
parser.add_argument('--vg', type=int, default=320)
parser.add_argument('--tail', type=str, default='mean')
parser.add_argument('--vq_tail', type=str, default='vq')
parser.add_argument('--mode', type=str, default='both', help='mean | vq | both')
args = parser.parse_args()

assert args.mode in ['mean','vq','both'], "Undefined Mode Selection"
extract_vq_index = args.mode in ['vq, 'both']
extract_mean_vector = args.mode in ['mean', 'both']

upstream = args.upstream
feat = args.tail
f_dim = args.dim
vg = args.vg


# === DATALOADER ===

dataloaders = {}

session_num = [1,2,3,4,5]
#session_num  = []

for except_session in range(len(session_num)):
    sessions = deepcopy(session_num)
    sessions.pop(except_session)
    dataset, collate_fn = load_dataset(
        data_name= 'iemocap',
        root= '/home/data/IEMOCAP',
        utterance_type= None,
        ext= 'pt',
        feature_path_tag= f'_{upstream}_{feat}',
        vq_path_tag= f'_{upstream}_vq' if extract_vq_index else None,
        final_classes= ["neu", "hap", "ang", "sad"],
        sessions = sessions,
        get_collate_fn=True)
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        num_workers=1,
        pin_memory=False,
        collate_fn=collate_fn,
        shuffle=False
    )

    dataloaders[f'iemocap_{except_session}'] = dict(train=dataloader, val=None, test=None)

kwargs_list = {
    "speechcommands": dict(
        root= 'data',
        folder_in_archive= f'SpeechCommands_{upstream}_{feat}',
        vq_folder= f'SpeechCommands_{upstream}_vq' if extract_vq_index else None,
        url= 'speech_commands_v0.02',
        ext= 'pt', get_collate_fn=True),
    "fluent": dict(
        data_name= 'fluent',
        root= f'/home/fluent_speech_commands/fluent_speech_commands_dataset_{upstream}_{feat}',
        vq_root= f'/home/fluent_speech_commands/fluent_speech_commands_dataset_{upstream}_vq' if extract_vq_index else None,
        ext= 'pt', get_collate_fn=True),
    "voxceleb": dict(
        data_name= 'voxceleb',
        root= f'data/voxceleb1_{upstream}_{feat}',
        vq_root= f'data/voxceleb1_{upstream}_vq' if extract_vq_index else None,
        url= 'iden_split.txt',
        ext= 'pt', get_collate_fn=True),
}

for data_name in kwargs_list:
    print("Load", data_name)
    collate_fn = load_dataset(**kwargs_list[data_name], subset='training')[1]
    dataloaders[data_name] = dict(
        train = DataLoader(
            dataset=load_dataset(**kwargs_list[data_name], subset='training')[0],
            batch_size=1,
            num_workers=1,
            pin_memory=False,
            collate_fn=collate_fn,
            shuffle=False
        ),
        val = DataLoader(
            dataset=load_dataset(**kwargs_list[data_name], subset='validation')[0],
            batch_size=1,
            num_workers=1,
            pin_memory=False,
            collate_fn=collate_fn,
            shuffle=False
        ),
        test = DataLoader(
            dataset=load_dataset(**kwargs_list[data_name], subset='testing')[0],
            batch_size=1,
            num_workers=1,
            pin_memory=False,
            collate_fn=collate_fn,
            shuffle=False
        )
    )

# === GET STATS ===
for data_key, _dataloaders in dataloaders.items():
    if (extract_mean_vector and os.path.exists(f'models/stats/{data_key}/mean_{upstream}_{feat}.pt')) \
    or (extract_vq_index and os.path.exists(f'models/stats/{data_key}/freq_{upstream}.pt')):
        print(f"models/stats/{data_key} Exists")
        continue
    print(data_key)
    train_dataloader, val_dataloader, test_dataloader = _dataloaders.values()

    token_dict = dict()

    vectors = list()
    vectorSum = torch.zeros(f_dim)

    # === Gather Frequency of Train & Validation Sets ===
    for dataloader in [train_dataloader, val_dataloader]:
        if dataloader is None: continue
        for batch in tqdm.tqdm(dataloader):
            if extract_vq_index:
                features, _, vq_index = batch[:-1]
            else:
                features, _ = batch[:-1]
            
            if extract_vq_index:
                # === save number of VQ index ===
                vq_index = vq_index[0].tolist() # This enormously accelarates the groupby calculation.
                for vq_index_ in vq_index:
                    # === Count index number ===
                    token_dict[tuple(vq_index_)] = token_dict.get(tuple(vq_index_), 0) + 1
            if extract_mean_vector:
                # === save features for mean calculation ===
                feature = features[0, -1] # last layer
                feature = feature.mean(0) # avg pool
                vectors.append(feature)
                vectorSum += feature

    if extract_vq_index:
        # === Map frequency to tensor (320 x 320) ===
        fields = torch.zeros((vg, vg))

        for key in token_dict.keys():
            key = list(map(int, key))
            fields[key[0], key[1]] = fields[key[0], key[1]] + token_dict[tuple(key)]

        torch.save(fields, f'models/stats/{data_key}/freq_{upstream}.pt')

    if extract_mean_vector:
        # === Calculate mean and covariances
        ## === Get Mean Vector ===    
        denominator = (len(train_dataloader) + len(val_dataloader)) if val_dataloader is not None else len(train_dataloader)
        mean_vector = vectorSum / denominator
        
        ## === Get Whiten Factor ===
        E = torch.stack(vectors)
        E_m = (E-mean_vector)
        Cov = torch.mm(E_m.T, E_m)
        U, S, V = torch.svd(Cov)
        D = torch.diag(1/torch.sqrt(S))
        whiten_factor = torch.mm(U,D)

        torch.save(mean_vector, f'models/stats/{data_key}/mean_{upstream}_{feat}.pt')
        torch.save(whiten_factor, f'models/stats/{data_key}/white_{upstream}_{feat}.pt')
        
