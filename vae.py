from __future__ import division
import numpy as np
import theano
import theano.tensor as T

from util import floatX, concat
from nnet import nnet, tanh_layer, sigmoid_layer


def init_params(Nx, Nz, encoder_hdims, decoder_hdims):
    def init_tensor(shape, name=None):
        return theano.shared(
            floatX(np.random.normal(size=shape) * 1e-2),
            borrow=True, name=name)

    def init_layer(shape):
        return init_tensor(shape), init_tensor(shape[1])

    def init_encoder(Nx, hdims, Nz):
        dims = [Nx] + hdims
        nnet_params = [init_layer(shape) for shape in zip(dims[:-1], dims[1:])]
        W_mu, b_mu = init_layer((hdims[-1], Nz))
        W_sigma, b_sigma = init_layer((hdims[-1], Nz))
        return nnet_params + [(W_mu, b_mu), (W_sigma, b_sigma)]

    def init_decoder(Nz, hdims, Nx):
        dims = [Nz] + hdims + [Nx]
        return [init_layer(shape) for shape in zip(dims[:-1], dims[1:])]

    encoder_params = init_encoder(Nx, encoder_hdims, Nz)
    decoder_params = init_decoder(Nz, decoder_hdims, Nx)

    return encoder_params, decoder_params, concat((encoder_params, decoder_params))


def encoder(X, encoder_params):
    'a neural net with tanh layers until the final layer,'
    'which generates mu and log_sigma separately'

    nnet_params, ((W_mu, b_mu), (W_sigma, b_sigma)) = \
        encoder_params[:-2], encoder_params[-2:]
    h = nnet(tanh_layer(W, b) for W, b in nnet_params)(X)
    mu = T.dot(h, W_mu) + b_mu
    log_sigma = 0.5 * (T.dot(h, W_sigma) + b_sigma)

    return mu, log_sigma


def decoder(Z, decoder_params):
    'a neural net with tanh layers until the final sigmoid layer'

    nnet_params, (W_out, b_out) = decoder_params[:-1], decoder_params[-1]
    h = nnet(tanh_layer(W, b) for W, b in nnet_params)(Z)
    Y = sigmoid_layer(W_out, b_out)(h)

    return Y


def vae_objective(X, encoder_params, decoder_params, M, L):
    z_dim = decoder_params[0].get_value().shape[0]

    def sample_z(mu, log_sigma):
        eps = srng.normal((M, z_dim), dtype=theano.config.floatX)
        return mu + T.exp(log_sigma) * eps

    def score_sample(Y):
        return -T.nnet.binary_crossentropy(X, Y)

    mu, log_sigma = encoder(X, encoder_params)

    kl_to_prior = 0.5 * T.sum(1. + 2.*log_sigma - mu**2. - T.exp(2.*log_sigma))
    logpxz = sum(score_sample(decode(sample_z(mu, log_sigma)))
                 for l in xrange(L)) / floatX(L)

    return kl_to_prior + logpxz
