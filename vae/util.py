import numpy as np
import theano
import logging
from collections import Iterable
from itertools import chain
from functools import wraps
from inspect import getcallargs, getargspec
from types import FunctionType
from operator import methodcaller


def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)


def shared_zeros_like(a):
    return theano.shared(
        np.zeros_like(a.get_value(), dtype=theano.config.floatX))


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def concat(lst):
    return list(chain(*lst))


def flatten(l):
    # NOTE: theano.sandbox.cuda.var.CudaNdarraySharedVariable is an instance of
    # collections.Iterable even though it doesn't support iteration!
    from theano.sandbox.cuda.var import CudaNdarraySharedVariable as CNSV

    if isinstance(l, Iterable) and not isinstance(l, CNSV):
        return [y for x in l for y in flatten(x)]
    else:
        return [l]


def argprint(f):
    def fstr(o):
        return '{}()'.format(o.__name__) if isinstance(o, FunctionType) \
            else str(o)

    @wraps(f)
    def wrapped(*args, **kwargs):
        bindings = getcallargs(f, *args, **kwargs)
        argspec = getargspec(f)
        arglist = ', '.join(
            '{}={}'.format(arg, fstr(bindings[arg])) for arg in argspec.args)
        logging.info('{}({})'.format(f.__name__, arglist))
        return f(*args, **kwargs)
    return wrapped


def reshape_square(a):
    sidelen = int(np.sqrt(a.shape[0]))
    return a.reshape(sidelen,sidelen)


def treemap(f,l):
    if isinstance(l, (list,tuple)):
        return [treemap(f,_) for _ in l]
    else:
        return f(l)


def get_ndarrays(params):
    return treemap(methodcaller('get_value'), params)
