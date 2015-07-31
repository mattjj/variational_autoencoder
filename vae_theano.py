import numpy as np
from numpy.random import permutation
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from time import time

from util import argprint
from load import load_mice
from vae import init_params, vae_objective
from optimization import sgd, adagrad, rmsprop, adadelta, adam, \
    momentum_sgd, nesterov


if __name__ == '__main__':
    np.random.seed(0)
    srng = RandomStreams(seed=1)

    N = 500000  # 750k is about the memory limit on 3GB GPU
    z_dim = 10
    h_dim = 200

    x_dim, trX = load_mice(N)
    encoder_params, decoder_params, all_params = \
        make_params(x_dim, [200], z_dim, [200])

    @argprint
    def fit(num_epochs, minibatch_size, optimizer, L=1):
        X = T.fmatrices('X')
        cost = -vae_objective(X, encoder_params, decoder_params, minibatch_size, L)
        updates = optimizer(cost, all_params)

        index = T.lscalar()
        train = theano.function(
            inputs=[index], outputs=cost, updates=updates,
            givens={X: trX[index*minibatch_size:(index+1)*minibatch_size]})
        num_batches = N // minibatch_size

        tic = time()
        for i in xrange(num_epochs):
            print sum(train(bidx) for bidx in permutation(num_batches)) / N
            print_W4()
        ellapsed = time() - tic
        print '{} sec per update, {} sec total\n'.format(ellapsed / N, ellapsed)

    def print_W4():
        s = np.linalg.svd(decoder_params[0].get_value())[1]
        print s[np.argsort(-s)]

    print_W4()
    fit(1, 20, adam(1e-4), callback=print_W4)
    fit(10, 20, adam(5e-4), callback=print_W4)
    fit(100, 1000, adam(1e-4), callback=print_W4)
