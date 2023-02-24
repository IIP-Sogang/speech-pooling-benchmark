import os
import os.path as osp
import argparse

TASKS = ['KWS','IC','ER','SI']
FEATUREEXTRACTOR = ['wav2vec2_base', 'xlsr_300m']

if __name__ == "__main__":
    ## Parse arguments
    parser = argparse.ArgumentParser(description = "Run a whole training sequence.")

    parser.add_argument('-c', '--config', type=str,   default='./configs/tasks/avgpool.yaml',   help='Config YAML file')
    parser.add_argument('-x', '--exp',     type=str,   default='exp1',   help='name dir')
    parser.add_argument('-p', '--profiler', type=str, default=None)
    parser.add_argument('-m', '--mode', type=str, default='train')

    args = parser.parse_args()

    with open(args.config, "r") as f:
        model_config_scripts = f.read()
        # model_config = yaml.safe_load(f)
    
    import subprocess
    # extractor loop
    for extractor in FEATUREEXTRACTOR:
        print(f"extractor : {extractor}")
        #  task loop
        for task in TASKS:
            print(f"task : {task}")
            task = task.lower()
            with open(f'configs/tasks/{task}.yaml', "r") as f:
                task_config_scripts = f.read()
            with open(f'configs/data/{task}_{extractor}.yaml', "r") as f:
                data_config_scripts = f.read()
        
            # Generate configs for training
            exp_path = f'exp/{task}/{args.exp}'
            if not osp.exists(exp_path):
                os.makedirs(exp_path)

            if args.mode=='train':
                with open(f'{exp_path}/{task}_{extractor}.yaml', 'w') as f:
                    f.write(f'default_root_dir: {exp_path}/{task}_{extractor}\n')
                    f.write(data_config_scripts+'\n')
                    f.write(task_config_scripts+'\n')                    
                    f.write(model_config_scripts)
            elif args.mode=='test':
                pass

            command = f'python main.py --config {exp_path}/{task}_{extractor}.yaml --mode {args.mode}'
            subprocess.run(command.split())