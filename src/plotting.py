import numpy as np
import matplotlib.pyplot as plt


def pdf_plot(x, fx, axis=None):
    if axis is None:
        axis = plt
    inds = np.argsort(x, axis=0).ravel()
    return axis.plot(x[inds], fx[inds])


def hist_plot(x, bins=None, axis=None):
    if axis is None:
        axis = plt
    return axis.hist(x, bins=bins, density=True)