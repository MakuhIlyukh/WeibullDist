import torch


def nll(ps):
    """ Negative log-likelihood loss """
    return -torch.mean(torch.log(ps))
