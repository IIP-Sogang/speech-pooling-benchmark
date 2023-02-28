
from typing import List, Literal, Tuple, Union
import pandas as pd

from dataloader.utils import load_dataset
from models.aggregate import VQWeightedAvgPool, SimpleAvgPool, VQAvgSuperPool, ProbWeightedAvgPool, SoftDecayPooling, WhiteningBERT, VQSqueezedAvgPool

import torch
from torch.utils.data import Dataset, DataLoader
from dataloader.utils import load_dataset


##========================
## Load DataLoader
##========================
def load_extracted_feature_dict(dir_tag:str='1_12', vq_dir_tag:str='vq')->dict:    
    """
    Dict
     - Dataset #1
         - train
         - val
         - test
     - Dataset #2
    Example Usage
    dataloader = load_extracted_feature_dict(..., ...)
    dataloader['speechcommands']['train'] <- This is dataloader
    """

    dataloaders = {}

    kwargs_list = {
        "speechcommands": dict(
            root= 'data',
            folder_in_archive= f'SpeechCommands_{dir_tag}',
            vq_folder= f'SpeechCommands_{vq_dir_tag}',
            url= 'speech_commands_v0.02',
            ext= 'pt', get_collate_fn=True),
        "fluent": dict(
            data_name= 'fluent',
            root= f'/home/fluent_speech_commands/fluent_speech_commands_dataset_{dir_tag}',
            vq_root= f'/home/fluent_speech_commands/fluent_speech_commands_dataset_{vq_dir_tag}',
            ext= 'pt', get_collate_fn=True),
        "voxceleb": dict(
            data_name= 'voxceleb',
            root= f'/home/voxceleb/voxceleb1_{dir_tag}',
            vq_root= f'/home/voxceleb/voxceleb1_{vq_dir_tag}',
            url= 'iden_split.txt',
            ext= 'pt', get_collate_fn=True),
    }

    for data_name in kwargs_list:
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

    from copy import deepcopy

    session_num = [] #[1,2,3,4,5]

    for except_session in range(len(session_num)):
        sessions = deepcopy(session_num)
        excluded_session_num = sessions.pop(except_session)
        trainset, collate_fn = load_dataset(
            data_name= 'iemocap',
            root= '/home/data/IEMOCAP', #'/home/nas4/DB/IEMOCAP', #
            utterance_type= None,
            ext= 'pt',
            feature_path_tag= f'_{dir_tag}',
            vq_path_tag= f'_{vq_dir_tag}',
            final_classes= ["neu", "hap", "ang", "sad"],
            sessions = sessions,
            get_collate_fn=True)
        
        testset, collate_fn = load_dataset(
            data_name= 'iemocap',
            root= '/home/data/IEMOCAP', # '/home/nas4/DB/IEMOCAP', # 
            utterance_type= None,
            ext= 'pt',
            feature_path_tag= f'_{dir_tag}',
            vq_path_tag= f'_{vq_dir_tag}',
            final_classes= ["neu", "hap", "ang", "sad"],
            sessions = [excluded_session_num],
            get_collate_fn=True)
        
        trainloader = DataLoader(
            dataset=trainset,
            batch_size=1,
            num_workers=1,
            pin_memory=False,
            collate_fn=collate_fn,
            shuffle=False
        )
        testloader = DataLoader(
            dataset=testset,
            batch_size=1,
            num_workers=1,
            pin_memory=False,
            collate_fn=collate_fn,
            shuffle=False
        )

        dataloaders[f'iemocap_{except_session}'] = dict(train=trainloader, val=None, test=testloader)

    return dataloaders

##========================
## Load Models Probpool
##========================
def load_prob_models(data_tag:str='fluent', feature_tag:str='wav2vec2_base', probs:list=[0.0001, 0.001, 0.01]):
    prob_models = {}
    for prob in probs:
        prob_models[f"prob_{prob}"] = ProbWeightedAvgPool(a=prob, freq_path=f'models/stats/{data_tag}/freq_{feature_tag}.pt')
    return prob_models

## =====================
## Load Model
## =====================
def load_model_dict(data_tags:List[str]= list(), feature_tag:str = 'wav2vec2_base', tail:str='')->Tuple[dict, dict]:
    models = dict(
        avg = SimpleAvgPool(),
        softdecay = SoftDecayPooling(),
        vq_ex = VQWeightedAvgPool(shrink='exact'),
        vq_or = VQWeightedAvgPool(shrink='or'),
        vq_sq = VQSqueezedAvgPool(),
        vq_chain = VQAvgSuperPool()
    )
    
    data_depedent_models = {}
    for data_tag in data_tags:
        data_depedent_models[data_tag] = {}
        data_depedent_models[data_tag]['white'] = WhiteningBERT([-1], True, f'./models/stats/{data_tag}/mean_{feature_tag}{"_"+tail if tail else ""}.pt', 
                                                                f'./models/stats/{data_tag}/white_{feature_tag}{"_"+tail if tail else ""}.pt', normalize='L2')
        prob_models = load_prob_models(data_tag, feature_tag, probs=[0.00001, 0.0001, 0.001, 0.01, 0.1])
        data_depedent_models[data_tag].update(prob_models)
        
    return models, data_depedent_models


def train(dataloaders:dict, models:dict, data_dependent_models:dict, 
          option:Literal['angular', 'euclidean', 'manhattan', 'hamming', 'dot']='angular', 
          head:str='wav2vec2_base', dir:str='.'):
    import os
    import tqdm
    import numpy as np
    from copy import deepcopy
    from annoy import AnnoyIndex

    if not os.path.exists(dir): os.makedirs(dir)

    for task in dataloaders.keys():
        dataloader = dataloaders[task]['train']
        
        _models = deepcopy(models)
        _models.update(data_dependent_models[task])
        
        f_dim = dataloader.dataset[0][0].size(-1) # 768
        
        annoy_dict = {key:(AnnoyIndex(f_dim, option), list()) for key in _models.keys()}

        for i, batch in tqdm.tqdm(enumerate(dataloader)):
            feature, length, vq, label = batch
            # feature, length, label = batch
            # vq = None

            for key in _models.keys():
                outputs = _models[key](feature, length, vq).squeeze()

                annoy_dict[key][0].add_item(i, outputs)
                annoy_dict[key][1].append(label.item())

        for key in _models.keys():
            annoy_dict[key][0].build(10)
            annoy_dict[key][0].save(f"{dir}/{head}_{task}_{key}_tree.ann")
            np.save(f"{dir}/{head}_{task}_{key}_label", np.array(annoy_dict[key][1]))


def test(dataloaders:dict, models:dict, data_dependent_models:dict, option:str='angular', head:str='wav2vec2_base', log_file:str='', dir:str='.'):
    import os
    import tqdm
    import numpy as np
    from copy import deepcopy
    from annoy import AnnoyIndex

    fs = open(f"{dir}/{log_file}", 'a')

    for task in dataloaders.keys():
        dataloader = dataloaders[task]['test']
        _models = deepcopy(models)
        _models.update(data_dependent_models[task])
        f_dim = dataloader.dataset[0][0].size(-1) # 768
        for key in _models.keys():
            print(f'{head}_{task}_{key} Testing Start')
            an_idx = AnnoyIndex(f_dim, 'angular')
            if not os.path.exists(f'{dir}/{head}_{task}_{key}_tree.ann'): 
                print(f'{dir}/{head}_{task}_{key}_tree.ann Does not exist. Pass it.')
                continue
            an_idx.load(f'{dir}/{head}_{task}_{key}_tree.ann')
            labels = np.load(f'{dir}/{head}_{task}_{key}_label.npy')
            
            total = 0
            correct = 0
            
            for i, batch in tqdm.tqdm(enumerate(dataloader)):
                feature, length, vq, label = batch
                # feature, length, label = batch
                # vq = None

                outputs = _models[key](feature, length, vq).squeeze()
                neareset_index = an_idx.get_nns_by_vector(outputs, 1)[0]
                if labels[neareset_index] == label:
                    correct += 1
                total += 1

            fs.write(f"{head},{task},{key},{correct},{total},{100*correct/total:.4f}%\n")
            print(f"{head}\t{task}\t{key}\t{correct}\t{total}\t{100*correct/total:.4f}%")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='results/unsupervised', help='Save result folder')
    parser.add_argument('--log', type=str, default='logs.csv')
    parser.add_argument('--metric', type=str, default='angular')
    parser.add_argument('--upstream', type=str, default='wav2vec2_base')
    parser.add_argument('--tail', type=str, default='mean')
    parser.add_argument('--vq_tail', type=str, default='vq')
    parser.add_argument('--mode', type=str, default='both', help='train | test | both')
    args = parser.parse_args()

    dataloaders = load_extracted_feature_dict('_'.join([args.upstream,args.tail]), '_'.join([args.upstream,args.vq_tail]))
    models, data_dependent_models = load_model_dict(dataloaders.keys(), args.upstream, tail=args.tail)

    if args.mode == 'test': pass
    else : train(dataloaders, models, data_dependent_models, option=args.metric, head=args.upstream, dir=args.dir)
    
    if args.mode=='train':pass
    else: test(dataloaders, models, data_dependent_models, option=args.metric, head=args.upstream, log_file=args.log, dir=args.dir)