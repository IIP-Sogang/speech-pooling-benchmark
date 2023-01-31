import pytorch_lightning as pl
import torch


class SpeechModel(pl.LightningModule):
    def __init__(self, model, loss_function, optimizer, scheduler, **kwargs):
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


    def training_step(self, batch, batch_idx):
        x, y = batch
        # preprocess
        
        # inference
        y_hat = self.model(x)

        # post processing

        # calculate loss
        loss = self.loss_function(y_hat, y)

        # Logging to TensorBoard
        self.log("loss", loss, on_epoch= True, prog_bar=True, logger=True, sync_dist=True)

        return loss

    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_function(y_hat, y)
        self.log("test_loss", loss,  on_epoch= True, prog_bar=True, logger=True, sync_dist=True)

        correct = (y_hat.argmax(1) == y).type(torch.float).sum().item()
        size = x.shape[0]
        return {'correct': correct, 'size': size}

    def test_epoch_end(self, test_step_outputs):
        correct_score = sum([dic['correct'] for dic in test_step_outputs])
        total_size = sum([dic['size'] for dic in test_step_outputs])
        acc = correct_score/total_size

        self.log("test_ACC", acc * 100, on_epoch = True, prog_bar=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_function(y_hat, y)
        self.log("val_loss", loss,  on_epoch= True, prog_bar=True, logger=True, sync_dist=True)

        correct = (y_hat.argmax(1) == y).type(torch.float).sum().item()
        size = x.shape[0]
        return {'correct': correct, 'size': size}

    def validation_epoch_end(self, validation_step_outputs):
        correct_score = sum([dic['correct'] for dic in validation_step_outputs])
        total_size = sum([dic['size'] for dic in validation_step_outputs])
        acc = correct_score/total_size

        self.log("val_ACC", acc * 100, on_epoch = True, prog_bar=True, sync_dist=True)

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