import numpy as np
from numpy.random import permutation
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from time import time

from util import argprint
from load import load_mice
from vae import make_gaussian_fitter
from optimization import sgd, adagrad, rmsprop, adadelta, adam, \
    momentum_sgd, nesterov


if __name__ == '__main__':
    np.random.seed(2)

    N = 750000  # 750k is about the memory limit on 3GB GPU
    trX = load_mice(N)

    encoder_params, decoder_params, fit = \
        make_gaussian_fitter(trX, 10, [25], [25])

    fit(1,  20, 1, adam(1e-6))
    fit(3,  20, 1, adam(1e-5))
    fit(9,  20, 1, adam(1e-5))
    fit(27, 20, 1, adam(1e-5))
    fit(1000, 5000, 1, adam(1e-6))

    # TODO call viz code as we go
    # TODO add plotting of training curves
