import os
import os.path as osp
import argparse

TASKS = ['KWS','IC','ER_0','ER_1','ER_2','ER_3','ER_4']#,'SI']
DATASETS = {
    'kws':'speechcommands',
    'ic':'fluent',
    'er_0':'iemocap_0',
    'er_1':'iemocap_1',
    'er_2':'iemocap_2',
    'er_3':'iemocap_3',
    'er_4':'iemocap_4',
    'si':'voxceleb'
}
FEATUREEXTRACTOR = ['wav2vec2_base']

def update_stats_config_script(script:str, data_key:str='speechcommands', upstream:str='wav2vec2_base', feat:str=''):
    tail = f"{upstream}{'_'+feat if feat else ''}"
    print(f"Load stats of {tail}")
    script = script.replace("freq_path: null", f"freq_path: models/stats/{data_key}/freq_{upstream}.pt")
    script = script.replace("mean_vector_path: null", f"mean_vector_path: models/stats/{data_key}/mean_{tail}.pt")
    script = script.replace("whiten_factor_path: null", f"whiten_factor_path: models/stats/{data_key}/white_{tail}.pt")
    return script

if __name__ == "__main__":
    ## Parse arguments
    parser = argparse.ArgumentParser(description = "Run a whole training sequence.")

    parser.add_argument('-c', '--config', type=str,   default='./configs/tasks/avgpool.yaml',   help='Config YAML file')
    parser.add_argument('-s', '--exp_dir',     type=str,   default='exp',   help='dir')
    parser.add_argument('-x', '--exp_num',     type=str,   default='exp_1',   help='name exp')
    parser.add_argument('-p', '--profiler', type=str, default=None)
    parser.add_argument('-t', '--task', type=str, default=None)
    parser.add_argument('-m', '--mode', type=str, default='train')

    args = parser.parse_args()

    if args.task is not None:
        TASKS = [args.task]
    with open(args.config, "r") as f:
        model_config_scripts = f.read()

    import subprocess
    # extractor loop
    for extractor in FEATUREEXTRACTOR:
        print(f"extractor : {extractor}")
        #  task loop
        for task in TASKS:
            print(f"task : {task}")
            task = task.lower()
            with open(f'configs/data/{task}_{extractor}.yaml', "r") as f:
                data_config_scripts = f.read()
            _model_config_scripts = update_stats_config_script(model_config_scripts,
                                                              data_key=DATASETS[task],
                                                              upstream=extractor,
                                                              feat='mean')
            _task = 'er' if 'er' in task else task
            with open(f'configs/tasks/{_task}.yaml', "r") as f:
                task_config_scripts = f.read()
            
            # Generate configs for training
            exp_path = f'{args.exp_dir}/{task}/{args.exp_num}'
            if not osp.exists(exp_path):
                os.makedirs(exp_path)

            if args.mode=='train':
                with open(f'{exp_path}/{task}_{extractor}.yaml', 'w') as f:
                    f.write(f'default_root_dir: {exp_path}/{task}_{extractor}\n')
                    f.write(data_config_scripts+'\n')
                    f.write(task_config_scripts+'\n')              
                    f.write(_model_config_scripts)
            elif args.mode=='test':
                pass

            command = f'python main.py --config {exp_path}/{task}_{extractor}.yaml --mode {args.mode}'
            subprocess.run(command.split())