#! /usr/bin/python
# -*- encoding: utf-8 -*-
# https://github.com/clovaai/voxceleb_trainer

import torch

def Optimizer(parameters, lr, weight_decay, **kwargs):

	print('Initialised SGD optimizer')

	return torch.optim.SGD(parameters, lr = lr, momentum = 0.9, weight_decay=weight_decay)
