import abc
from tqdm import tqdm
from time import perf_counter_ns
from dataclasses import dataclass

import torch

from src.models import (
    Optimized_WM, Manual_GD_WM, EM_WM_TORCH, WM)


def get_optimizer(opt_name):
    if opt_name == "sgd":
        return torch.optim.SGD
    elif opt_name == "adam":
        return torch.optim.Adam
    else:
        raise ValueError("Unknown optimizer!")


def compute_batch_size(batch_size_or_ratio, full_size):
    if isinstance(batch_size_or_ratio, float):
            batch_size_or_ratio = int(full_size * batch_size_or_ratio)
    elif isinstance(batch_size_or_ratio, int):
        pass
    else:
        raise ValueError("batch_size_or_ratio must be int or float <= 1.0")
    return batch_size_or_ratio


@dataclass
class IterInfo:
    iter_time: int
    loss: float
    metric: float


class BaseTrainer:
    @abc.abstractmethod
    def step(self, X):
        raise NotImplementedError("This is abstract method!")
    
    def train(self, X, y_true, n_epochs, batch_size,
              loss_fn, metric_fn, loss_prefix, metric_prefix):
        loss_name = loss_prefix + "_loss"
        metric_name = metric_prefix + "_score"
        time_name = "iter_time"
        # TODO: add stop on plateu
        batch_size = compute_batch_size(batch_size, X.shape[0])
        cum_time = 0
        numbers = []
        tqdm_bar = tqdm(range(n_epochs), position=1, leave=False)
        for epoch in tqdm_bar:
            start_iter_time = perf_counter_ns()
            for batch_start in range(0, X.shape[0], batch_size):
                self.step(X[batch_start : batch_start + batch_size])
            end_iter_time = perf_counter_ns()

            iter_time = end_iter_time - start_iter_time
            cum_time += iter_time
            
            with torch.no_grad():
                dens = self.model(X)
                loss = loss_fn(dens).item()
                metric_score = metric_fn(dens, y_true).item()

            tqdm_bar.set_postfix({
                loss_name: loss,
                metric_name: metric_score})
            
            numbers.append(IterInfo(
                iter_time=iter_time,
                loss=loss,
                metric=metric_score
            ))

        return numbers


class GD_Trainer(BaseTrainer):
    def __init__(self, m, opt_name, lr, k_init, lmd_init, q_init, loss_fn):
        self.model = WM(
            m=m, k_init=k_init, lmd_init=lmd_init, q_init=q_init)
        self.optimizer = get_optimizer(opt_name)(
            self.model.parameters(), lr=lr)
        self.loss_fn = loss_fn

    def step(self, X):
        self.optimizer.zero_grad()
        pdf = self.model(X)
        loss = self.loss_fn(pdf)
        loss.backward()
        self.optimizer.step()
        return pdf, loss


class OptimizedGD_Trainer(BaseTrainer):
    def __init__(self, m, opt_name, lr, k_init, lmd_init, q_init, loss_fn):
        self.model = Optimized_WM(
            m=m, k_init=k_init, lmd_init=lmd_init, q_init=q_init)
        self.optimizer = get_optimizer(opt_name)(
            self.model.parameters(), lr=lr)
        self.loss_fn = loss_fn

    def step(self, X):
        self.optimizer.zero_grad()
        pdf = self.model(X)
        loss = self.loss_fn(pdf)
        loss.backward()
        self.optimizer.step()
        return pdf, loss


class ManualGD_Trainer(BaseTrainer):
    def __init__(self, m, opt_name, lr, k_init, lmd_init, q_init, c, eps):
        self.model = Manual_GD_WM(
            m=m, k_init=k_init, lmd_init=lmd_init, q_init=q_init, c=c, eps=eps)
        self.optimizer = get_optimizer(opt_name)(
            self.model.parameters(), lr=lr)
    
    def step(self, X):
        self.optimizer.zero_grad()
        q_grad, k_grad, lmd_grad = self.model.compute_grad(X)
        self.model.q_w.grad = q_grad
        self.model.k_w.grad = k_grad
        self.model.lmd_w.grad = lmd_grad
        self.optimizer.step()
        # TODO: return pdf and loss
        return None, None


class EM_Trainer(BaseTrainer):
    def __init__(self, m, k_init, lmd_init, q_init, max_newton_iter):
        self.model = EM_WM_TORCH(m, k_init=k_init, lmd_init=lmd_init, q_init=q_init)
        self.max_newton_iter = max_newton_iter

    def step(self, X):
        self.model.step(X, max_newton_iter=self.max_newton_iter)
        # TODO: return pdf and loss
        return None, None
