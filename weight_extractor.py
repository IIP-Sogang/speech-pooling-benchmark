import importlib
import argparse

import torch
from tqdm import tqdm
import numpy as np
import yaml
from torch.utils.data import DataLoader
from torchmetrics.functional import kl_divergence
import pandas as pd

# ===============================================
# set dataset for KWS & IC
# ===============================================

config_dict = {
    'ic':{
        'avgpool': './exp_xlsr/ic/xlsr_avgpool/ic_xlsr.yaml',
        'sap': './exp_xlsr/ic/xlsr_sap/ic_xlsr.yaml',
        'vqlocal': './exp_xlsr/ic/xlsr_vqlocal/ic_xlsr.yaml',
        'vqglobal': './exp_xlsr/ic/xlsr_vqglobal/ic_xlsr.yaml',
        'probpool_001': './exp_xlsr/ic/xlsr_probpool_001/ic_xlsr.yaml',
        'probpool_00001': './exp_xlsr/ic/xlsr_probpool_00001/ic_xlsr.yaml',
    },
    'kws':{
        'avgpool': './exp_xlsr/kws/xlsr_avgpool/kws_xlsr.yaml',
        'sap': './exp_xlsr/kws/xlsr_sap/kws_xlsr.yaml',
        'vqlocal': './exp_xlsr/kws/xlsr_vqlocal/kws_xlsr.yaml',
        'vqglobal': './exp_xlsr/kws/xlsr_vqglobal/kws_xlsr.yaml',
        'probpool_001': './exp_xlsr/kws/xlsr_probpool_001/kws_xlsr.yaml',
        'probpool_00001': './exp_xlsr/kws/xlsr_probpool_00001/kws_xlsr.yaml',
    }

}

def extract(config, sap_config):
    
    from dataloader.utils import load_dataset
    if sap_config.get('validation_dataset_config', False):
        sap_config['test_loader_config'] = sap_config['validation_loader_config']
        sap_config['test_dataset_config'] = sap_config['validation_dataset_config']
    test_dataset, test_collate_fn = load_dataset(**sap_config['test_dataset_config'], get_collate_fn=True)
    print(f"test_data_len: {len(test_dataset)}")

    test_dataloader = DataLoader(
            dataset = test_dataset,
            batch_size=1,
            num_workers=config['num_workers'],
            pin_memory=False,
            collate_fn=test_collate_fn,
            **config.get('test_loader_config',{})
        )
    
    # ⚡⚡ 2. Load Model
    model = importlib.import_module('models.tasks').__getattribute__(config['model'])
    model =  model(**config['model_config'])

    # avg_model = importlib.import_module('models.tasks').__getattribute__(avg_config['model'])
    # avg_model =  avg_model(**avg_config['model_config'])

    sap_model = importlib.import_module('models.tasks').__getattribute__(sap_config['model'])
    sap_model =  sap_model(**sap_config['model_config'])

    # Load checkpoint
    ckpt_path = config['resume_checkpoint'] 
    ckpt = torch.load(ckpt_path, map_location='cpu')
    modified_ckpt = modify_key(ckpt)
    model.load_state_dict(modified_ckpt)

    # avg_ckpt_path = avg_config['resume_checkpoint']
    # avg_ckpt = torch.load(avg_ckpt_path, map_location='cpu')
    # avg_modified_ckpt = modify_key(avg_ckpt)
    # avg_model.load_state_dict(avg_modified_ckpt)

    sap_ckpt_path = sap_config['resume_checkpoint']
    sap_ckpt = torch.load(sap_ckpt_path, map_location='cpu')
    sap_modified_ckpt = modify_key(sap_ckpt)
    sap_model.load_state_dict(sap_modified_ckpt)


    # compute KL divergence
    avg_list, sap_list, as_list = compute_kl_divergence_loop(model, sap_model, test_dataloader, config['devices'])
    
    return avg_list, sap_list, as_list


def modify_key(ckpt):
    modified_ckpt = {}
    for key, value in ckpt['state_dict'].items():
        if key.startswith('model'):
            key = key.replace('model.', '')
            modified_ckpt[key] = value
        else:
            modified_ckpt[key] = value
    return modified_ckpt



def compute_kl_divergence_loop(model, sap_model, test_dataloader, device):
    # https://torchmetrics.readthedocs.io/en/stable/regression/kl_divergence.html
    
    sap_model.eval()
    model.eval()

    kl_avg_list = []
    kl_sap_list = []
    kl_avg_sap_list = []
    with torch.no_grad():
        for _, batch in enumerate(tqdm(test_dataloader)):
            x = batch[:-1] # (Input, Input_lengths) or (Input, Input_lengths, Input_vq_indices)
            length = x[1]
            w = model.head.get_weight(*x)
            w_avg = torch.ones_like(w) / length
            w_sap = sap_model.head.get_weight(x[0], x[1])

            assert torch.allclose(w.sum(), torch.tensor([1.0])) , "Sum of w is not 1"
            assert torch.allclose(w_avg.sum(), torch.tensor([1.0])) , "Sum of w_avg is not 1"
            assert torch.allclose(w_sap.sum(), torch.tensor([1.0])), "Sum of w_sap is not 1"

            
            kl_avg = kl_divergence(w, w_avg)
            kl_sap = kl_divergence(w, w_sap)
            kl_avg_sap = kl_divergence(w_avg, w_sap)

            kl_avg_list.append(kl_avg)
            kl_sap_list.append(kl_sap)
            kl_avg_sap_list.append(kl_avg_sap)

    
    return kl_avg_list, kl_sap_list, kl_avg_sap_list


if __name__ == "__main__":
    ## Parse arguments
    parser = argparse.ArgumentParser(description = "Speaker verification with sequential module")

    parser.add_argument('--model',         type=str,   default='vqlocal',   help='vqlocal | vqglobal | probpool_001 | probpool_00001')
    parser.add_argument('--task',         type=str,   default='ic',   help='ic | kws')
    parser.add_argument('--all',        action="store_true", help="True or False")

    args = parser.parse_args()

    #Load yaml
    # avg_config_path = config_dict.get(args.task).get('avgpool')
    if not args.all:        
        sap_config_path = config_dict.get(args.task).get('sap')
        config_path = config_dict.get(args.task).get(args.model)

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # with open(avg_config_path, "r") as f:
        #     avg_config = yaml.safe_load(f)

        with open(sap_config_path, "r") as f:
            sap_config = yaml.safe_load(f)

        kl_avg_list, kl_sap_list, kl_avg_sap_list = extract(config, sap_config)

        print(f"avg_val : {np.mean(kl_avg_list)}, sap_val : {np.mean(kl_sap_list)}, as_val : {np.mean(kl_avg_sap_list)}")

    else:
        config_d = config_dict.get(args.task)
        df = pd.DataFrame(columns=['task', 'model', 'avg_val', 'sap_val', 'as_val'])

        for key_model in config_d.keys():
            if key_model == 'sap':
                continue
            elif key_model == 'avgpool':
                continue

            print(f"model : {key_model}")

            config_path = config_d.get(key_model)
            sap_config_path = config_d.get('sap')

            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

            with open(sap_config_path, "r") as f:
                sap_config = yaml.safe_load(f)

            kl_avg_list, kl_sap_list, kl_avg_sap_list = extract(config, sap_config)

            df_temp = pd.DataFrame([[args.task, key_model, np.mean(kl_avg_list), np.mean(kl_sap_list), np.mean(kl_avg_sap_list)]], columns=['task', 'model', 'avg_val', 'sap_val', 'as_val'])
            df = pd.concat([df, df_temp])



            print(f"avg_val : {np.mean(kl_avg_list)}, sap_val : {np.mean(kl_sap_list)}, as_val : {np.mean(kl_avg_sap_list)}")

        print(df)
        save_path = f"./{args.task}_kl_divergence_comparison.csv"
        df.to_csv(save_path, index=False)


