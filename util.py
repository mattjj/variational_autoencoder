import numpy as np
import theano
from itertools import chain
from functools import wraps
from inspect import getcallargs, getargspec


def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)


def shared_zeros_like(a):
    return theano.shared(
        np.zeros_like(a.get_value(), dtype=theano.config.floatX))


def concat(lst):
    return list(chain(*lst))


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def argprint(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        bindings = getcallargs(f, *args, **kwargs)
        argspec = getargspec(f)
        arglist = ', '.join(
            '{}={}'.format(arg, bindings[arg]) for arg in argspec.args)
        print '{}({})'.format(f.__name__, arglist)
        return f(*args, **kwargs)
    return wrapped

