import pytorch_lightning as pl
import torch
from models.tasks.utils import vector_quantizer, _shrink


def acc_step(y_hat, y):
    correct = (y_hat.argmax(1) == y).type(torch.float).sum().item()
    size = y.shape[0]
    return {'correct': correct, 'size': size}

def acc_calculate(step_outputs:dict, name:str='val'):
    correct_score = sum([dic['correct'] for dic in step_outputs])
    total_size = sum([dic['size'] for dic in step_outputs])
    acc = correct_score/total_size
    return name+'_ACC', acc*100

def slot_acc_step(y_hat, y):
    # slot1: 6 /slot2: 14 /slot3: 4
    slot_start = 0
    slot_sizes = [6, 14, 4]
    preds = list()
    for size in slot_sizes:
        preds.append(y_hat[:,slot_start:slot_start+size].argmax(-1))
        slot_start += size
    preds = torch.stack(preds, dim=-1)
    correct = (preds == y).prod(1).sum().item()
    size = y.shape[0]
    return {'correct': correct, 'size': size}

def eer_step(y_hat, y):
    return {'label': y, 'sim':y_hat}

def eer_calculate(step_outputs:dict, name:str='val'):
    import numpy as np
    from sklearn.metrics import roc_curve
    from scipy.optimize import brentq
    from scipy.interpolate import interp1d
    labels = torch.cat([dic['label'] for dic in step_outputs])
    scores = torch.cat([dic['sim'] for dic in step_outputs])
    labels = labels.detach().cpu()
    scores = scores.detach().cpu()
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    return name+'_EER', eer*100

def test_loss_switch(func, metric:str):
    # Switch loss function, only for evaluation
    from loss.cosine_emb import loss_function as cos_loss
    
    if metric == "eer":
        return cos_loss
    else:
        return func

MetricFuncs = dict(
    acc=dict(
        step_func=acc_step,
        epoch_func=acc_calculate
    ),
    slot_acc=dict(
        step_func=slot_acc_step,
        epoch_func=acc_calculate
    ),
    eer=dict(
        step_func=eer_step,
        epoch_func=eer_calculate
    )
)


class SpeechModel(pl.LightningModule):
    def __init__(self, model, loss_function, optimizer, scheduler, metric='acc', method = 'conv_feature', **kwargs):
        super().__init__()
        # ⚡ model
        self.model = model
        print(self.model)

        # ⚡ loss 
        self.loss_function = loss_function

        # ⚡ optimizer
        self.optimizer = optimizer

        # ⚡ scheduler
        self.scheduler = scheduler # **kwargs: **config['scheduler_config']

        # save hyperparameters
        self.save_hyperparameters(ignore=['model'])

        #⚡⚡⚡ debugging - print input output layer ⚡⚡⚡
        sample_size = tuple(map(int, kwargs['sample_input_size'].split())) if kwargs.get('sample_input_size', False) else (64,1,28,28)
        self.example_input_array = torch.randn(sample_size)

        # custom
        self.metric = metric
        self.method = method # 'conv_feature' or 'transformer_feature' or 'wav'


    def training_step(self, batch, batch_idx):

        x = batch[:-1] # (Input, Input_lengths) or (Input, Input_vq_indices, Input_lengths)
        y = batch[-1]
        # preprocess
        
        # inference
        y_hat = self.model(*x)

        # post processing

        # calculate loss
        loss = self.loss_function(y_hat, y)
        # Logging to TensorBoard
        self.log("loss", loss, on_epoch= True, prog_bar=True, logger=True, sync_dist=True)

        return loss

    def test_step(self, batch, batch_idx):
        # this is the test loop
        x = batch[:-1]
        y = batch[-1]

        y_hat = self.model(*x)
        loss_function = test_loss_switch(self.loss_function, self.metric)
        loss = loss_function(y_hat, y)
        self.log("test_loss", loss,  on_epoch= True, prog_bar=True, logger=True, sync_dist=True)

        return MetricFuncs[self.metric]['step_func'](y_hat, y)

    def test_epoch_end(self, test_step_outputs):
        name, value = MetricFuncs[self.metric]['epoch_func'](test_step_outputs, name='test')
        self.log(name, value, on_epoch = True, prog_bar=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x = batch[:-1]
        y = batch[-1]

        y_hat = self.model(*x)
        loss_function = test_loss_switch(self.loss_function, self.metric)
        loss = loss_function(y_hat, y)
        self.log("val_loss", loss,  on_epoch= True, prog_bar=True, logger=True, sync_dist=True)

        return MetricFuncs[self.metric]['step_func'](y_hat, y)

    def validation_epoch_end(self, validation_step_outputs):
        name, value = MetricFuncs[self.metric]['epoch_func'](validation_step_outputs, name='val')
        self.log(name, value, on_epoch = True, prog_bar=True, sync_dist=True)

    def forward(self, x):
        y_hat = x
        return y_hat

    def configure_optimizers(self):
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "monitor": "val_loss",
                "interval": self.scheduler.interval,
                "frequency": self.scheduler.frequency,
                "name": 'lr_log'
            },
        }