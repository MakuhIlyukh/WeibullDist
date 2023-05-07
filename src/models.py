# %%
from typing import Iterable

import torch
from torch import nn
from torch.nn.utils import parametrize
from tqdm import tqdm

from src.initializers import (
    k_initialize, lmd_initialize, q_initialize)


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

        self.k_w = k_initialize(m, k_init)
        
        # scale parameters
        # TODO: strongly positive
        self.lmd_w = torch.nn.Parameter(
            torch.empty(m, dtype=torch.float64, requires_grad=True))
        parametrize.register_parametrization(
            self, 'lmd_w', SquareParametrization())
        
        self.lmd_w = lmd_initialize(m, lmd_init)
        
        # components probs
        self.q_w = torch.nn.Parameter(
            torch.empty(m, dtype=torch.float64, requires_grad=True))
        parametrize.register_parametrization(
            self, 'q_w', SoftmaxParametrization(c))
        
        self.q_w = q_initialize(m, q_init)
        
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


# ???: nn.Module or empty?
class EM_WM_TORCH(nn.Module):
    def __init__(self, m, k_init, lmd_init, q_init, c=0):
        super().__init__()

        # shape parameters
        # TODO: strongly positive
        # self.k_w = torch.nn.Parameter(
            # torch.empty(m, dtype=torch.float64, requires_grad=False))

        self.k_w = k_initialize(m, k_init)
        
        # scale parameters
        # TODO: strongly positive
        # self.lmd_w = torch.nn.Parameter(
            # torch.empty(m, dtype=torch.float64, requires_grad=False))
        
        self.lmd_w = k_initialize(m, lmd_init)
        
        # components probs
        # self.q_w = torch.nn.Parameter(
            # torch.empty(m, dtype=torch.float64, requires_grad=False))
        
        self.q_w = q_initialize(m, q_init)
        
        # constants
        self.m = m

    def cond_probs(self, x):
        with parametrize.cached():
            # TODO: maybe it's better to create variable 1_div_lmd
            x_div_lmd = x / self.lmd_w
            s = (self.q_w
                    * self.k_w / self.lmd_w
                    * x_div_lmd**(self.k_w - 1)
                    * torch.exp(-(x_div_lmd)**self.k_w))
        return s

    def pdf(self, X):
        return self.cond_probs(X).sum(axis=1, keepdim=True)
    
    def fit(self, X, max_iter, max_newton_iter):
        with torch.no_grad():
            t = 0
            lnX = torch.log(X)
            n = X.shape[0]
            converged = False
            bar = tqdm(total=100)
            while t < max_iter and not converged:
                # NOTE: порядок важен
                cprobs = self.cond_probs(X)
                # TODO: Должно ли присутствовать q?
                cpz = cprobs / torch.sum(cprobs, axis=1, keepdim=True)
                cpz_sum = torch.sum(cpz, axis=0)
                # ???: нет ли опечатки в формуле???
                cpz_lnX = cpz*lnX
                A_r = torch.sum(cpz_lnX, axis=0) / cpz_sum
                self.q_w = cpz_sum / n
                r = 0
                newton_converged = False
                # ???: должна ли инициализация другими значениями, а не значениями на прошлой итерации?
                K_r = self.k_w
                while r < max_newton_iter and not newton_converged:
                    B_r = cpz * X**K_r
                    C_r = B_r * lnX
                    D_r = C_r * lnX
                    B_r = torch.sum(B_r, axis=0)
                    C_r = torch.sum(C_r, axis=0)
                    D_r = torch.sum(D_r, axis=0)
                    C_div_B = C_r / B_r
                    one_div_K = 1 / K_r
                    old_K_r = K_r
                    # ???: maybe better one_div_k / K_R
                    K_r = K_r + (A_r + one_div_K - C_div_B) / (one_div_K * one_div_K + D_r / B_r - C_div_B / B_r)
                    r += 1
                    # TODO: newton_convergence flag
                    newton_converged = False
                # K_r назначается с предпоследней итерации, чтобы не перещитывать B_r
                self.k_w = old_K_r
                self.lmd_w = (B_r / cpz_sum)**one_div_K
                t += 1
                # TODO: convergence flag
                converged = False
                bar.update(1)
            bar.close()


class Optimized_WM(torch.nn.Module):
    def __init__(self, m, k_init, lmd_init, q_init, c=0):
        super().__init__()

        # shape parameters
        # TODO: strongly positive
        self.k_w = torch.nn.Parameter(
            torch.empty(m, dtype=torch.float64, requires_grad=True))
        parametrize.register_parametrization(
            self, 'k_w', SquareParametrization())

        self.k_w = k_initialize(m, k_init)
        
        # scale parameters
        # TODO: strongly positive
        self.lmd_w = torch.nn.Parameter(
            torch.empty(m, dtype=torch.float64, requires_grad=True))
        parametrize.register_parametrization(
            self, 'lmd_w', SquareParametrization())
        
        self.lmd_w = lmd_initialize(m, lmd_init)
        
        # components probs
        self.q_w = torch.nn.Parameter(
            torch.empty(m, dtype=torch.float64, requires_grad=True))
        parametrize.register_parametrization(
            self, 'q_w', SoftmaxParametrization(c))
        
        self.q_w = q_initialize(m, q_init)
        
        # constants
        self.m = m
    
    def forward(self, x):
        with parametrize.cached():
            x_div_lmd = x / self.lmd_w
            pow_k = torch.exp(self.k_w*torch.log(x_div_lmd))
            s = ((self.q_w * self.k_w)  # brackets are important
                    / x
                    * pow_k
                    * torch.exp(-pow_k))
        return torch.sum(s, axis=1, keepdim=True)


class Manual_GD_WM(torch.nn.Module):
    def __init__(self, m, k_init, lmd_init, q_init, c=0, eps=1e-6):
        super().__init__()

        # shape parameters
        self.k_w = torch.nn.Parameter(
            torch.empty(m, dtype=torch.float64, requires_grad=True))

        self.k_w.copy_(k_initialize(m, k_init, manual_parametrization=True, eps=1e-6))
        
        # scale parameters
        self.lmd_w = torch.nn.Parameter(
            torch.empty(m, dtype=torch.float64, requires_grad=True))
        
        self.lmd_w.copy_(lmd_initialize(m, lmd_init, manual_parametrization=True, eps=1e-6))
        
        # components probs
        self.q_w = torch.nn.Parameter(
            torch.empty(m, dtype=torch.float64, requires_grad=True))
        
        self.q_w.copy_(q_initialize(m, q_init, manual_parametrization=True, c=1e-6))

        # constants
        self.m = m
        self.c = c
        self.eps = eps

    def compute_grad(self, x):
        with torch.no_grad():
            q_w = self.q_w
            k_w = self.k_w
            lmd_w = self.lmd_w

            q = nn.functional.softmax(q_w, dim=-1, dtype=torch.float64)
            k = k_w * k_w + self.eps
            l = lmd_w * lmd_w + self.eps

            x_div_l = x / l
            log_pow_k  = torch.log(x_div_l)*k
            pow_k = torch.exp(log_pow_k) 
            expn = torch.exp(-pow_k)
            expn_pow_div = expn * pow_k / x
            components = expn_pow_div * (k * q)  # braces are important!
            pdf = components.sum(axis=1, keepdim=True)
            q_grad = expn_pow_div * (-k) / pdf  # braces are important!
            pow_k_m_1 = pow_k - 1
            l_grad = q_grad*pow_k_m_1
            k_grad = expn_pow_div * (log_pow_k * pow_k_m_1 - 1) / pdf

            q_jac = -torch.outer(q, q) + torch.diag(q)
            return q_grad.mean(axis=0) @ q_jac, k_grad.mean(axis=0)*q*2*k_w, l_grad.mean(axis=0) * (k * q / l) * 2*lmd_w
    
    def forward(self, x):
        q = nn.functional.softmax(self.q_w, dim=-1, dtype=torch.float64)
        k = self.k_w * self.k_w + self.eps
        l = self.lmd_w * self.lmd_w + self.eps

        x_div_lmd = x / l
        pow_k = torch.exp(k*torch.log(x_div_lmd))
        s = ((q * k)  # brackets are important
                / x
                * pow_k
                * torch.exp(-pow_k))
        return torch.sum(s, axis=1, keepdim=True)