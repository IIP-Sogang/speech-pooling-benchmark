import os
import sys
import tqdm
import argparse
import torch

from dataloader.utils import load_dataset, VoxCelebDataset
from dataloader.dataset import DATA_LIST
from models.feats_extractor import Wav2VecXLSR03BExtractor, Wav2VecExtractor, VQWav2VecExtractor, VQWav2VecXLSR03BExtractor, load_extractor


def main(data_name:str, root:str='data', new_root:str='data2', url:str=None, subset:str=None, tag:str=None, layer_ids:str="1_12", ext_type = 'Wav2VecXLSR03BExtractor', **kwargs):
    assert data_name in DATA_LIST, f"{data_name} IS NOT EXISTING DATASET!!"

    if tag is None: tag = "_"+layer_ids
    layer_ids = list(map(int, layer_ids.split('_')))

    DEVICE = 'cuda'if torch.cuda.is_available() else 'cpu'
    dataset = load_dataset(root=root, data_name=data_name, url=url, subset=subset, ext='wav') # should return (an audio array, sampling rate)
    # dataset = load_dataset(root=root, data_name=data_name, subset=subset, ext='wav')

    # select baseline model(feature extractor)
    extractor = load_extractor(ext_type = ext_type)
    extractor = extractor.to(DEVICE)

    extractor.eval()
    for i in tqdm.tqdm(range(len(dataset))):
        new_path = dataset.generate_feature_path(i, new_root=new_root, tag=tag)
        waveform, sr = dataset[i]
        waveform = waveform.to(DEVICE)
        with torch.no_grad():
            features = extractor.extract(waveform, sr)
            features = [feature.detach().cpu() for feature in features]
        features = torch.cat(features, axis=0)
        
        # Save features from selected layers
        save_features = list()
        for layer_id in layer_ids:
            save_features.append(features[layer_id])
        save_features = torch.stack(save_features)
        torch.save(save_features, new_path)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description = "pre feature extraction module")

    parser.add_argument('--data_name',         type=str,   default='fluent',   help='iemocap | fluent | voxceleb | speechcommands')

    # KWS
    # parser.add_argument('--root',         type=str,   default='/home/nas3/user/jeonko/spc_embedding/data',   \
                        # help='/home/nas4/DB/IEMOCAP | /home/nas4/DB/fluent_speech_commands/fluent_speech_commands_dataset | /home/nas4/DB/speaker_verification/data/voxceleb1 | /home/nas3/user/jeonko/spc_embedding/data')

    # ER
    # parser.add_argument('--root',         type=str,   default='/home/nas4/DB/IEMOCAP',   \
                        # help='/home/nas4/DB/IEMOCAP | /home/nas4/DB/fluent_speech_commands/fluent_speech_commands_dataset | /home/nas4/DB/speaker_verification/data/voxceleb1 | /home/nas3/user/jeonko/spc_embedding/data')
    # IC
    parser.add_argument('--root',         type=str,   default='/home/nas4/DB/fluent_speech_commands/fluent_speech_commands_dataset',   \
                        help='/home/nas4/DB/IEMOCAP | /home/nas4/DB/fluent_speech_commands/fluent_speech_commands_dataset | /home/nas4/DB/speaker_verification/data/voxceleb1 | /home/nas3/user/jeonko/spc_embedding/data')

    # VoxCeleb
    # parser.add_argument('--root',         type=str,   default='/home/nas4/DB/speaker_verification/data/voxceleb1',   \
                        # help='/home/nas4/DB/IEMOCAP | /home/nas4/DB/fluent_speech_commands/fluent_speech_commands_dataset | /home/nas4/DB/speaker_verification/data/voxceleb1 | /home/nas3/user/jeonko/spc_embedding/data')
    # KWS
    # parser.add_argument('--new_root',         type=str,   default='/home/nas3/user/jeonko/spc_embedding/data',   help='/home/nas4/DB/IEMOCAP/IEMOCAP | /home/nas4/DB/fluent_speech_commands/fluent_speech_commands_dataset | | /home/nas3/user/jeonko/spc_embedding/data(no use) ')
    
    # ER
    parser.add_argument('--new_root',         type=str,   default='/home/nas4/DB/fluent_speech_commands/fluent_speech_commands_dataset',   help='/home/nas4/DB/IEMOCAP/IEMOCAP | \
                        /home/nas4/DB/fluent_speech_commands/fluent_speech_commands_dataset | | /home/nas3/user/jeonko/spc_embedding/data(no use) ')

    parser.add_argument('--url',         type=str,   default='speech_commands_v0.02',   help='X | X | vrfy_split.txt| speech_commands_v0.02')
    parser.add_argument('--subset',         type=str,   default='validation',   help='["train", "valid", "test"]')
    parser.add_argument('--tag',         type=str,   default='_wav2vec2_large_1_24',   help='_xlsr_feat_1_24 | _xlsr_1_24 | _xlsr_vq | _hubert_large_1_24')
    # parser.add_argument('--tag',         type=str,   default='_wav2vec2_large_1_24',   help='_xlsr_feat_1_24 | _xlsr_1_24 | _xlsr_vq | _hubert_large_1_24')
    # parser.add_argument('--tag',         type=str,   default='_hubert_large_1_24',   help='_xlsr_feat_1_24 | _xlsr_1_24 | _xlsr_vq')
    parser.add_argument('--layer_ids',         type=str,   default='1_-1',   help='1_12 | 1_-1')
    # parser.add_argument('--ext_type',         type=str,   default='HubertLarge',   help='select feature extractor')
    parser.add_argument('--ext_type',         type=str,   default='Wav2VeLargeExtractor',   help='select feature extractor')

    args = parser.parse_args()
    print(args)
    main(**vars(args))