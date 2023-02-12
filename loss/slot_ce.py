import torch.nn.functional as F

def loss_function(y_hat, target):
    # slot1: 6 /slot2: 14 /slot3: 4
    slot_start = 0
    slot_sizes = [6, 14, 4]
    intent_loss = 0.
    for i, size in enumerate(slot_sizes):
        intent_loss += F.cross_entropy(y_hat[:,slot_start:slot_start+size], target[:,i])
        slot_start += size
    return intent_loss