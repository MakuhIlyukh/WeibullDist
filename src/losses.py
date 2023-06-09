import torch


def nll(ps):
    """ Negative log-likelihood loss """
    return -torch.mean(torch.log(ps))


def components_entropy(q):
    return -torch.sum(q*torch.log(q))


def BIC(n, m, ll):
    return 3*m*torch.log(n) - 2*ll