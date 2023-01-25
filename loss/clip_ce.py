import torch.nn.functional as F

def loss_function(y_hat, target):
    return F.cross_entropy(y_hat, target)