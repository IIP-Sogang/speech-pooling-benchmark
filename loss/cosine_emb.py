import torch
import torch.nn.functional as F

def loss_function(y_hat, target, margin=0):
    batch_size = y_hat.size(0)
    loss = torch.tensor(0.).to(y_hat.device)
    for i, target_ in enumerate(target):
        if target_ == 1:
            loss += 1 - y_hat[i]
        else:
            loss += torch.maximum(torch.tensor(0).to(y_hat.device), y_hat[i] - margin)
    return loss/batch_size