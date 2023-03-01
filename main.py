import argparse
import yaml
import torch
import sys
import importlib
import warnings
warnings.filterwarnings('ignore')

from torch.utils.data import DataLoader

import lightning_fabric as lf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.profiler import SimpleProfiler, AdvancedProfiler, PyTorchProfiler, XLAProfiler

from engine import SpeechModel

PROFILERS = {
    "simple": SimpleProfiler,
    "advanced": AdvancedProfiler,
    "pytorch": PyTorchProfiler,
    "xla": XLAProfiler,}


def train(config):
    # ⚡⚡ 1. Set 'Dataset', 'DataLoader'  
    from dataloader.utils import load_dataset
    training_dataset, train_collate_fn = load_dataset(**config['training_dataset_config'], get_collate_fn=True)
    if config.get('validation_dataset_config', False):
        config['test_loader_config'] = config['validation_loader_config']
        config['test_dataset_config'] = config['validation_dataset_config']
    test_dataset, test_collate_fn = load_dataset(**config['test_dataset_config'], get_collate_fn=True)

    train_dataloader = DataLoader(
            dataset = training_dataset,
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            pin_memory=False,
            collate_fn=train_collate_fn,
            **config.get('train_loader_config',{})
        )

    test_dataloader = DataLoader(
            dataset = test_dataset,
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            pin_memory=False,
            collate_fn=test_collate_fn,
            **config.get('test_loader_config',{})
        )

    # ⚡⚡ 2. Set 'Model', 'Loss', 'Optimizer', 'Scheduler'
    model = importlib.import_module('models.tasks').__getattribute__(config['model'])
    model =  model(**config['model_config'])

    optimizer = importlib.import_module("optimizer." + config['optimizer']).__getattribute__("Optimizer")
    optimizer = optimizer(model.parameters(), **config['optimizer_config'])

    loss_function = importlib.import_module("loss." + config['loss']).__getattribute__("loss_function")

    scheduler = importlib.import_module("scheduler." + config['scheduler']).__getattribute__("Scheduler")
    scheduler = scheduler(optimizer, **config['scheduler_config'])


    # ⚡⚡  3. Set 'engine' for training/validation and 'Trainer' 
    model = SpeechModel(model = model, optimizer=optimizer, loss_function=loss_function, scheduler=scheduler, **config.get('task_config', dict())) 


    # ⚡⚡ 4. Init callbacks
    checkpoint_callback = ModelCheckpoint(
        **config['checkpoint_config']
    )
    lr_callback = LearningRateMonitor()


    # ⚡⚡ Logger
    from lightning_utilities.core.imports import RequirementCache
    _TENSORBOARD_AVAILABLE = RequirementCache("tensorboard")
    _TENSORBOARDX_AVAILABLE = RequirementCache("tensorboardX")
    if not (_TENSORBOARD_AVAILABLE or _TENSORBOARDX_AVAILABLE):
        print("Warning : Tensorboard is not available, CSV logger will be used.")

    # ⚡⚡ Profiler
    profiler = PROFILERS[config['profiler']](dirpath=config['default_root_dir'], filename='profile_report') if config.get('profiler',None) else None

    # ⚡⚡ 5. LightningModule
    trainer = pl.Trainer(
        deterministic=True, # Might make your system slower, but ensures reproducibility.
        default_root_dir = config['default_root_dir'], #
        devices = config['devices'], #
        val_check_interval = 1.0, # Check val every n train epochs.
        max_epochs = config['max_epoch'], #
        auto_lr_find = False, # ⚡⚡
        sync_batchnorm = True, # ⚡⚡
        callbacks = [checkpoint_callback, lr_callback], #
        accelerator = config['accelerator'], #
        num_sanity_val_steps = config['num_sanity_val_steps'], # Sanity check runs n batches of val before starting the training routine. This catches any bugs in your validation without having to wait for the first validation check. 
        replace_sampler_ddp = True, # ⚡⚡
        profiler = profiler,
    )

    # ⚡⚡ 6. Resume training
    if config['resume_checkpoint']  is not None:
        print("⚡")
        trainer.fit(model, train_dataloader, test_dataloader, ckpt_path=config['resume_checkpoint'])
        print(config['resume_checkpoint'] + "are loaded")
    else:
        print("⚡⚡")
        trainer.fit(model, train_dataloader, test_dataloader)
        print("no pre-trained weight are loaded")


def test(config):
    print("test")

    # sets seeds for numpy, torch and python.random.    
    lf.utilities.seed.seed_everything(seed = config['random_seed'])

    # ⚡⚡ 1. Set 'Dataset', 'DataLoader' 
    from dataloader.utils import load_dataset
    test_dataset, collate_fn = load_dataset(**config['test_dataset_config'], get_collate_fn=True)

    test_dataloader = DataLoader(
            dataset = test_dataset,
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            pin_memory=False,
            collate_fn=collate_fn,
            **config.get('test_loader_config',{})
        )

    # ⚡⚡ 2. Set 'Model', 'Loss', 'Optimizer', 'Scheduler'
    model = importlib.import_module('models.tasks').__getattribute__(config['model'])
    model =  model(**config['model_config'])

    optimizer = importlib.import_module("optimizer." + config['optimizer']).__getattribute__("Optimizer")
    optimizer = optimizer(model.parameters(), **config['optimizer_config'])

    loss_function = importlib.import_module("loss." + config['loss']).__getattribute__("loss_function")

    scheduler = importlib.import_module("scheduler." + config['scheduler']).__getattribute__("Scheduler")
    scheduler = scheduler(optimizer, **config['scheduler_config'])

    # ⚡⚡  3. Load model
    model = SpeechModel.load_from_checkpoint(model = model, optimizer=optimizer, loss_function=loss_function, scheduler=scheduler, checkpoint_path = config['resume_checkpoint'], **config.get('task_config', dict())) 

    # ⚡⚡ 4. LightningModule
    trainer = pl.Trainer(accelerator=config['accelerator'], gpus = config['devices'])

    trainer.test(model, dataloaders=test_dataloader)

    
if __name__ == "__main__":
    ## Parse arguments
    parser = argparse.ArgumentParser(description = "Speaker verification with sequential module")

    parser.add_argument('--config',         type=str,   default='./configs/mnist.yaml',   help='Config YAML file')
    parser.add_argument('--mode',         type=str,   default='train',   help='choose train/val/eval')

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)


    print(args)
    print(config)
    print('Python Version:', sys.version)
    print('PyTorch Version:', torch.__version__)
    print('Number of GPUs:', torch.cuda.device_count())

    if args.mode == "train":
        train(config)

    elif args.mode == "test":
        config['batch_size'] = 1
        test(config)

    # sets seeds for numpy, torch and python.random.
    