#! /usr/bin/python
# -*- encoding: utf-8 -*-

import warnings
import torch
from torch.optim.lr_scheduler import _LRScheduler

def Scheduler(optimizer, **kwargs):
    sche_fn = ConstantLRScheduler(optimizer, last_epoch=kwargs.get('last_epoch', -1))
    print('Initialized constant LR scheduler')
    return sche_fn

class ConstantLRScheduler(_LRScheduler):
    def __init__(self, optimizer, last_epoch=-1, verbose=False):
        super().__init__(optimizer, last_epoch, verbose)
        self.interval = 'epoch'
        self.frequency = 1

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        return [group['lr'] for group in self.optimizer.param_groups]
        
    def _get_closed_form_lr(self):
        return self.base_lrs