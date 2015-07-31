from __future__ import division
from theano import tensor as T


def nnet(*layers):
    return reduce(lambda f,g: lambda h: g(f(h)), layers)


def make_layer(activation):
    def layer(W, b):
        def apply(h):
            return activation(T.dot(h, W) + b)
        return apply
    return layer


tanh_layer = make_layer(T.tanh)
sigmoid_layer = make_layer(T.nnet.sigmoid)
linear_layer = make_layer(lambda h: h)
