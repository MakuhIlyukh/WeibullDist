""" Weibull Mixture Datasets """


import pickle

import numpy as np
import matplotlib.pyplot as plt


class WeibullMixtureSampler:
    def __init__(self, m, rnd_state, k_init, lmd_init, q_init):
        """

        :param m: number of components
        """
        if not callable(k_init):
            raise ValueError("k_init must be callable!")
        if not callable(lmd_init):
            raise ValueError("lmd_init must be callable!")
        if not callable(q_init):
            raise ValueError("q_init must be callable!")

        self._m = m
        self.rnd_state = rnd_state
        self._q = q_init(m, rnd_state)
        self._k = k_init(m, rnd_state)
        self._lmd = lmd_init(m, rnd_state)

    def sample(self, n):
        # TODO: Choose dtype
        X = np.empty((n, 1), dtype=np.float64)
        y = self.rnd_state.choice(self._m, p=self._q, size=n)
        for i, c in enumerate(y):
            X[i] = self._lmd[c] * self.rnd_state.weibull(self._k[c], size=1)
        return X, y

    def save(self, file_like):
        """ Saves mixture to file.

        :param file_like: for example open(filename, "wb")
        """
        pickle.dump(self, file_like)
    
    @staticmethod
    def load(file_like):
        """ Loads mixture from file.

        :param file_like: for example open(filename, "wb")
        """
        return pickle.load(file_like)

    def pdf(self, x):
        """ Only for non-negative values. """
        s = np.zeros((x.shape[0], 1), dtype=np.float64)
        for j in range(self._m):
            s += (self._q[j]
                  * self._k[j] / self._lmd[j]
                  * (x / self._lmd[j])**(self._k[j]-1)
                  * np.exp(-(x / self._lmd[j])**self._k[j]))
        return s
    
    @property
    def m(self):
        return self._m
    
    @property
    def q(self):
        return self._q
    
    @property
    def k(self):
        return self._k
    
    @property
    def lmd(self):
        return self._lmd


def save_dataset(X, y, file_like):
    """ Saves (X, y) to file. """
    pickle.dump((X, y), file_like)


def load_dataset(file_like):
    """ Loads (X, y) from file. """
    return pickle.load(file_like)
