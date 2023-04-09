import torch


def nll(ps):
    """ Negative log-likelihood loss """
    return -torch.sum(torch.log(ps))
