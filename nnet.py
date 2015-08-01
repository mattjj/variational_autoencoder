from __future__ import division
import numpy as np
import theano
import theano.tensor as T

from util import floatX


def make_layer(activation):
    def layer(W, b):
        def apply(h):
            return activation(T.dot(h, W) + b)
        return apply
    return layer


tanh_layer = make_layer(T.tanh)
sigmoid_layer = make_layer(T.nnet.sigmoid)
linear_layer = make_layer(lambda x: x)


def compose(layers):
    return reduce(lambda f,g: lambda h: g(f(h)), layers)


def init_tensor(shape, name=None):
    return theano.shared(
        floatX(1e-2 * np.random.normal(size=shape)),
        borrow=True, name=name)


def init_layer(shape):
    return init_tensor(shape), init_tensor(shape[1])
