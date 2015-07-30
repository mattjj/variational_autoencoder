import numpy as np
from numpy.random import permutation
import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from time import time
from itertools import chain

from util import floatX
from optimization import sgd, adagrad, rmsprop, adadelta, adam, \
    momentum_sgd, nesterov

np.random.seed(1)
srng = RandomStreams(seed=1)


def vae_objective(minibatch_size, X, W1, W2, W3, W4, W5, b1, b2, b3, b4, b5, L=1):
    h_encoder = T.tanh(T.dot(X, W1) + b1)
    mu_encoder = T.dot(h_encoder, W2) + b2
    log_sigma_encoder = 0.5 * (T.dot(h_encoder, W3) + b3)

    kl_to_prior = 0.5 * T.sum(
        1. + 2.*log_sigma_encoder - mu_encoder**2. - T.exp(2.*log_sigma_encoder))

    logpxz = 0.
    z_dim = W4.get_value().shape[0]
    for l in xrange(L):
        eps = srng.normal((minibatch_size, z_dim), dtype=theano.config.floatX)
        z = mu_encoder + T.exp(log_sigma_encoder) * eps
        h_decoder = T.tanh(T.dot(z, W4) + b4)
        Y = T.nnet.sigmoid(T.dot(h_decoder, W5) + b5)
        logpxz += -T.nnet.binary_crossentropy(Y,X).sum()

    return kl_to_prior + logpxz / floatX(L)


### initialization

def init_weights(shape,name=None):
    return theano.shared(
        floatX(np.random.randn(*shape) * 1e-2), borrow=True, name=name)


def make_params(Nx, Nh, Nz):
    weight_shapes = [(Nx, Nh),(Nh, Nz), (Nh, Nz), (Nz, Nh), (Nh, Nx)]
    bias_shapes = [(b,) for a, b in weight_shapes]
    return [init_weights(shape) for shape in chain(weight_shapes, bias_shapes)]


### loading data

def load_mice():
    data = np.load('data/images_for_vae.npy').astype(theano.config.floatX)[::2]
    data = np.random.permutation(data.reshape(data.shape[0], -1))
    data /= data.max()
    shared_data = theano.shared(floatX(data), borrow=True)
    return data.shape, shared_data


### running

if __name__ == '__main__':
    z_dim = 30
    h_dim = 400

    (N, x_dim), trX = load_mice()

    X = T.fmatrices('X')
    params = W1, W2, W3, W4, W5, b1, b2, b3, b4, b5 = \
        make_params(x_dim, h_dim, z_dim)

    def fit(num_epochs, minibatch_size, learning_rate):
        cost = -vae_objective(minibatch_size, X, *params)
        updates = adam(cost, params, learning_rate)

        index = T.lscalar()
        train = theano.function(
            inputs=[index], outputs=cost, updates=updates,
            givens={X: trX[index*minibatch_size:(index+1)*minibatch_size]})

        num_batches = N // minibatch_size

        print
        print 'num_epochs = {}'.format(num_epochs)
        print 'minibatch_size = {}'.format(minibatch_size)
        print 'learning_rate = {}'.format(learning_rate)
        print

        for i in xrange(num_epochs):
            tic = time()
            objective = sum(train(bidx) for bidx in permutation(num_batches)) / N
            print '{} {}'.format(time() - tic, objective)

    fit(250, 5000, 1e-3)
    fit(250, 5000, 1e-4)
