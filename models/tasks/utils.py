import torch
import torch.nn as nn



def vector_quantizer(x_tr, x_conv):
    # x_tr [BATCH, 2, TIME FRAME, DIM]
    # x_conv : [BATCH, TIME FRAME, 2]
    import pdb;pdb.set_trace()

    
    int_distance = x_conv[:, :-1, :] - x_conv[:, 1:, :] # [BATCH, TIME FRAME -1, 2]
    int_distance = int_distance.sum(-1) # [BATCH, TIME FRAME -1]
    x = torch.cat([int_distance[:,:1], int_distance], dim = 1) # [BATCH, TIME FRAME]

    # normalize
    x = nn.functional.normalize(x.float(), dim = 0) # [BATCH, TIME FRAME]

    x = x.unsqueeze(dim = 1)  # [BATCH, 1, TIME FRAME]
    x = x.unsqueeze(dim = 3) # [BATCH, 1, TIME FRAME, 1]
    
    x_tr = x_tr[:, -1, :] # select final transformer layer [BATCH, 1, TIME FRAME, DIM]

    x = x_tr * x
    return x # [BATCH, 1, TIME FRAME, DIM]
