import os
import os.path as osp
import argparse

TASKS = ['KWS','IC','ER','SI']

if __name__ == "__main__":
    ## Parse arguments
    parser = argparse.ArgumentParser(description = "Run a whole training sequence.")

    parser.add_argument('-c', '--config', type=str,   default='./configs/avgpool.yaml',   help='Config YAML file')
    parser.add_argument('-x', '--exp',     type=str,   default='exp1',   help='name dir')
    parser.add_argument('-p', '--profiler', type=str, default=None)
    parser.add_argument('-m', '--mode', type=str, default='train')

    args = parser.parse_args()

    with open(args.config, "r") as f:
        model_config_scripts = f.read()
        # model_config = yaml.safe_load(f)

    import subprocess
    for task in TASKS:
        task = task.lower()
        with open(f'configs/tasks/{task}.yaml', "r") as f:
            task_config_scripts = f.read()
    
        # Generate configs for training
        exp_path = f'exp/{task}/{args.exp}'
        if not osp.exists(exp_path):
            os.makedirs(exp_path)

        if args.mode=='train':
            # assert not osp.exists(f'exp/{task}/{args.exp}.yaml'), "Exp Results can be Overwritten"
            with open(f'exp/{task}/{args.exp}.yaml', 'w') as f:
                f.write(f'default_root_dir: {exp_path}\n')
                f.write(task_config_scripts+'\n')
                f.write(model_config_scripts)
        elif args.mode=='test':
            pass

        command = f'python main.py --config exp/{task}/{args.exp}.yaml --mode {args.mode}'
        subprocess.run(command.split())