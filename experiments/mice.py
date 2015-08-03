import numpy as np
from numpy.random import permutation
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from time import time

from util import argprint
from load import load_mice
from vae import init_gaussian_params, make_gaussian_objective
from optimization import sgd, adagrad, rmsprop, adadelta, adam, \
    momentum_sgd, nesterov


def make_fitter(trX, z_dim, encoder_hdims, decoder_hdims, callback=None):
    N, x_dim = trX.get_value().shape
    encoder_params, decoder_params, all_params = \
        init_gaussian_params(x_dim, z_dim, encoder_hdims, decoder_hdims)
    vlb = make_gaussian_objective(encoder_params, decoder_params)

    @argprint
    def fit(num_epochs, minibatch_size, L, optimizer):
        num_batches = N // minibatch_size

        X = T.matrix('X', dtype=theano.config.floatX)
        cost = -vlb(X, N, minibatch_size, L)
        updates = optimizer(cost, all_params)

        index = T.lscalar()
        train = theano.function(
            inputs=[index], outputs=cost, updates=updates,
            givens={X: trX[index*minibatch_size:(index+1)*minibatch_size]})

        tic = time()
        for i in xrange(num_epochs):
            costval = sum(train(bidx) for bidx in permutation(num_batches))
            print 'iter {:>4} of {:>4}: {:> .6}'.format(i+1, num_epochs, costval / N)
            if callback: callback()
        ellapsed = time() - tic
        print '{} sec per update, {} sec total\n'.format(ellapsed / N, ellapsed)

    return encoder_params, decoder_params, fit


if __name__ == '__main__':
    np.random.seed(2)

    N = 750000  # 750k is about the memory limit on 3GB GPU
    trX = load_mice(N)

    encoder_params, decoder_params, fit = make_fitter(trX, 10, [25], [25])

    fit(1,  20, 1, adam(1e-6))
    fit(3,  20, 1, adam(1e-5))
    fit(9,  20, 1, adam(1e-5))
    fit(27, 20, 1, adam(1e-5))
    fit(1000, 5000, 1, adam(1e-6))

    # TODO call viz code as we go
    # TODO add plotting of training curves
