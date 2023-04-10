# %%
from typing import Iterable

import torch
from torch import nn
from torch.nn.utils import parametrize
from tqdm import tqdm


class SoftmaxParametrization(torch.nn.Module):
    def __init__(self, c=0):
        super().__init__()
        # TODO: maybe it's better to cast to tensor 
        self.c = c

    def forward(self, X):
        return nn.functional.softmax(X, dim=-1)
    
    def right_inverse(self, Y):
        return torch.log(Y) + self.c


class SquareParametrization(torch.nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        # TODO: maybe it's better to cast to tensor
        self.eps = eps
    
    def forward(self, X):
        return X*X + self.eps
    
    def right_inverse(self, Y):
        return torch.sqrt(Y - self.eps)


class WM(torch.nn.Module):
    def __init__(self, m, k_init, lmd_init, q_init, c=0):
        super().__init__()

        # shape parameters
        # TODO: strongly positive
        self.k_w = torch.nn.Parameter(
            torch.empty(m, dtype=torch.float64, requires_grad=True))
        parametrize.register_parametrization(
            self, 'k_w', SquareParametrization())

        with torch.no_grad():
            if isinstance(k_init, str) and k_init == "random":
                self.k_w = (torch.rand(
                    m, dtype=torch.float64) + 1)
            elif callable(k_init):
                self.k_w = (k_init())
            else:
                raise ValueError("k_init must be callable or 'random'")
        
        # scale parameters
        # TODO: strongly positive
        self.lmd_w = torch.nn.Parameter(
            torch.empty(m, dtype=torch.float64, requires_grad=True))
        parametrize.register_parametrization(
            self, 'lmd_w', SquareParametrization())
        
        with torch.no_grad():
            if isinstance(lmd_init, str) and lmd_init == "random":
                self.lmd_w = (torch.rand(
                    m, dtype=torch.float64) + 1)
            elif callable(lmd_init):
                self.lmd_w = (lmd_init())
            else:
                raise ValueError("lmd_init must be callable or 'random'")
        
        # components probs
        self.q_w = torch.nn.Parameter(
            torch.empty(m, dtype=torch.float64, requires_grad=True))
        parametrize.register_parametrization(
            self, 'q_w', SoftmaxParametrization(c))
        
        with torch.no_grad():
            if isinstance(q_init, str):
                if q_init == 'dirichlet':
                    self.q_w = torch.distributions.Dirichlet(
                            concentration=torch.full(
                                (m,), 1.0, dtype=torch.float64)
                        ).sample()
                elif q_init == "1/m":
                    self.q_w = torch.full(
                        (m,), 1/m, dtype=torch.float64)
                else:
                    raise ValueError("q_init must be 'dirchlet', '1/k' or callable")
            elif callable(q_init):
                self.q_w = q_init()
            else:
                raise ValueError("q_init must be 'dirchlet', '1/k' or callable")
        
        # constants
        self.m = m
    
    def forward(self, x):
        with parametrize.cached():
            # TODO: maybe it's better to create variable 1_div_lmd
            x_div_lmd = x / self.lmd_w
            s = (self.q_w
                    * self.k_w / self.lmd_w
                    * x_div_lmd**(self.k_w - 1)
                    * torch.exp(-(x_div_lmd)**self.k_w))
        return torch.sum(s, axis=1, keepdim=True)