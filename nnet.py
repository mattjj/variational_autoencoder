from __future__ import division
import numpy as np
import theano
import theano.tensor as T

from util import floatX, sigmoid


### constructing and composing layers

def make_layer(dot, activation):
    def layer(W, b):
        def apply(h):
            return activation(dot(h, W) + b)
        return apply
    return layer


def compose(layers):
    return reduce(lambda f,g: lambda h: g(f(h)), layers, lambda x: x)


### initialization

def init_tensor(shape, name=None):
    return theano.shared(
        floatX(1e-2 * np.random.normal(size=shape)),
        borrow=True, name=name)


def init_layer(shape):
    return init_tensor(shape), init_tensor(shape[1])


### theano-backed layers

tanh_layer = make_layer(T.dot, T.tanh)
sigmoid_layer = make_layer(T.dot, T.nnet.sigmoid)
linear_layer = make_layer(T.dot, lambda x: x)


### numpy-backed layers

numpy_tanh_layer = make_layer(np.dot, np.tanh)
numpy_sigmoid_layer = make_layer(np.dot, sigmoid)
numpy_linear_layer = make_layer(np.dot, lambda x: x)
