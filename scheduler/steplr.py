#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
from torch.optim.lr_scheduler import StepLR
from optimizer.adam import Optimizer

def Scheduler(optimizer, step_size, gamma, **kwargs):
    sche_fn = StepLRScheduler(optimizer, step_size=step_size, gamma=gamma, last_epoch=kwargs.get('last_epoch', -1))
    print('Initialized step LR scheduler')
    return sche_fn

class StepLRScheduler(StepLR):
    def __init__(self, optimizer: Optimizer, step_size: int, gamma: float = ..., last_epoch: int = ...) -> None:
        super().__init__(optimizer, step_size, gamma, last_epoch)
        self.interval = 'epoch'
        self.frequency = 1