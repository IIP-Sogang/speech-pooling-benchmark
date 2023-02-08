import torch
import torch.nn as nn
import torch.nn.functional as F


# def loss_function(y_hat, labels, s=30.0, m=0.4, out_features=None):
#     '''
#     input shape (N, in_features)
#     '''
#     assert len(y_hat) == len(labels)
#     assert torch.min(labels) >= 0
#     # assert torch.max(labels) < out_features
#     wf = y_hat
#     numerator = s * (torch.diagonal(wf.transpose(0, 1)[labels]) - m)
#     excl = torch.cat([torch.cat((wf[i, :y], wf[i, y+1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
#     denominator = torch.exp(numerator) + torch.sum(torch.exp(s * excl), dim=1)
#     L = numerator - torch.log(denominator)
#     return -torch.mean(L)

def loss_function(costh, lb, s=30.0, m=0.4, out_features=None):
    assert costh.size()[0] == lb.size()[0]
    lb_view = lb.view(-1, 1)
    if lb_view.is_cuda: lb_view = lb_view.cpu()
    delt_costh = torch.zeros(costh.size()).scatter_(1, lb_view, m)
    if costh.is_cuda: delt_costh = delt_costh.cuda()
    costh_m = costh - delt_costh
    costh_m_s = s * costh_m
    loss = torch.nn.functional.cross_entropy(costh_m_s, lb, reduction='mean')
    return loss