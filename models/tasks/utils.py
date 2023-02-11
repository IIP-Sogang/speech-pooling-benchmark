import torch
import torch.nn as nn



def vector_quantizer(x_tr, x_conv):
    # x_tr [BATCH, 2, TIME FRAME, DIM]
    # x_conv : [BATCH, TIME FRAME, 2]
    
    int_distance = x_conv[:, :-1, :] - x_conv[:, 1:, :] # [BATCH, TIME FRAME -1, 2]
    int_distance = int_distance.sum(-1) # [BATCH, TIME FRAME -1]
    x = torch.cat([int_distance[:,:1], int_distance], dim = 1) # [BATCH, TIME FRAME]

    # normalize
    x = nn.functional.normalize(x.float(), dim = 0) # [BATCH, TIME FRAME]

    x = x.unsqueeze(dim = 1)  # [BATCH, 1, TIME FRAME]
    x = x.unsqueeze(dim = 3) # [BATCH, 1, TIME FRAME, 1]
    
    # x_tr = x_tr[:, -1, :, :] # select final transformer layer [BATCH, TIME FRAME, DIM]
    # x_tr = x_tr.unsqueeze(dim = 1) # [BATCH, 1, TIME FRAME, DIM]

    x = x_tr * x
        # x_tr : [BATCH, 2, TIME FRAME, DIM]
        # x: [BATCH, 1, TIME FRAME, 1]
        # x(@result): [BATCH, 2, TIME FRAME, DIM]
    return x # [BATCH, 2, TIME FRAME, DIM]

def _shrink(x_tr, x_conv, equality="exact"):
    """
    This function operates in the following sequence:

    1. Find padded token index
    2. Restore padded feature
    3. Calculate weight based on restored feature
    4. Multiply weight to transformer feature

    Args:
        x_tr : Transformer layer feature
        x_conv : Convolution layer feature
    Returns:
        x: weighted transformer feature       
    """
    def _exact(l, r):
        return (l[0].item() == r[0].item()) & (l[1].item() == r[1].item())

    def _partial(l, r):
        return (l[0].item() == r[0].item()) | (l[1].item() == r[1].item())
    
    def _find_fake_token(x_conv):
        """
        This function finds padded token index in batch of convolution features.
        Also, this function findns time frames before padded.

        Args:
            x_conv : padded convolution features
        Returns:
            fake_token_index_batch_list: list of padded token index in batch
            fake_token_index_time_list: list of time frames before padded        
        """

        fake_token_index_batch_list = []
        fake_token_index_time_list = []

        for idx_batch, x_conv_sample in enumerate(x_conv): # x_conv_sample: [TIME FRAME, 2]
            for idx_time, x_conv_sample_time in enumerate(x_conv_sample): # x_conv_sample_time: [2]

                # find fake token index
                if (x_conv_sample_time == torch.tensor([-1., -1.]).to(x_conv_sample_time.device)).sum().item() == 2:
                    fake_token_index_batch_list.append(idx_batch)
                    fake_token_index_time_list.append(idx_time)
                    break

        return fake_token_index_batch_list, fake_token_index_time_list
    
    MAX_TIME_FRAME = x_conv.shape[1]

    # ===================================
    # 1. find padded token index
    # ===================================
    fake_token_index_batch_list, fake_token_index_time_list = _find_fake_token(x_conv)

    # ===================================
    # 2. restore padded features
    # ===================================
    conv_feat_restored = []    

    for idx_batch, conv_feat in enumerate(x_conv):
        # if not padded
        if idx_batch not in fake_token_index_batch_list:
            conv_feat_restored.append(conv_feat)

        # elif padded
        elif idx_batch in fake_token_index_batch_list:

            idx = fake_token_index_batch_list.index(idx_batch) # find fake token index

            time_frame = fake_token_index_time_list[idx] # find time_frame

            assert idx_batch == fake_token_index_batch_list[idx], "check index of padded features"
            conv_feat_restored.append(conv_feat[:time_frame, :])

        else:
            raise ValueError("check index of padded features")
            
    # ⚡ print(conv_feat_restored) <- debugging point            

    # ===================================
    # 3. calculate weights
    # ===================================
    equality_func = {"exact": _exact, "partial": _partial}[equality]

    weights = torch.zeros_like(x_conv[:,:,-1], dtype = torch.float32) # [BATCH, TIME FRAME]

    for idx_batch, conv_feat in enumerate(conv_feat_restored): # idx_batch, [TIME FRAME, 2] in [BATCH, TIME FRAME, 2]
        i = 0
        time_dim = conv_feat.shape[0]
        weight = []
        cluster_count = 0

        while i < time_dim:
            cluster_size = 1

            while (i + cluster_size) < time_dim:
                is_eq = equality_func(conv_feat[i], conv_feat[i + cluster_size])
                if is_eq:
                    cluster_size += 1
                else:
                    break

            weight += [1 / cluster_size] * cluster_size
            i += cluster_size
            cluster_count += 1

        assert len(weight) == time_dim, "check time frame of conv_feat"

        weight = torch.FloatTensor(weight) / cluster_count # len(weight) : TIME FRAME

        weights[idx_batch,:time_dim] = weight # weights : [BATCH, TIME FRAME]

    # ⚡ print(weights) <- debugging point

    # ===================================
    # 4. Multiply weight to transformer feature
    # ===================================

    weights = weights.unsqueeze(dim = 1)  # [BATCH, 1, TIME FRAME]
    weights = weights.unsqueeze(dim = 3) # [BATCH, 1, TIME FRAME, 1]
    

    x = x_tr * weights # [BATCH, 2, TIME FRAME, DIM] * [BATCH, 1, TIME FRAME, 1] ⚡ <- debugging point


    return x