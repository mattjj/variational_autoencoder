import numpy as np
import theano
from itertools import chain


def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)


def shared_zeros_like(a):
    return theano.shared(
        np.zeros_like(a.get_value(), dtype=theano.config.floatX))


def concat(lst):
    return list(chain(*lst))


def sigmoid(x):
    return 1. / (1. + np.exp(-x))
