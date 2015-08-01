from __future__ import division
import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from util import floatX, flatten
from nnet import compose, tanh_layer, sigmoid_layer, linear_layer, init_layer

srng = RandomStreams(seed=1)


def init_params(Nx, Nz, encoder_hdims, decoder_hdims):
    'initialize variational autoencoder parameter lists as shared variables'

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

    return encoder_params, decoder_params, flatten((encoder_params, decoder_params))


def encoder(encoder_params):
    'a neural net with tanh layers until the final layer,'
    'which generates mu and log_sigmasq separately'

    nnet_params, ((W_mu, b_mu), (W_sigma, b_sigma)) = \
        encoder_params[:-2], encoder_params[-2:]
    nnet = compose(tanh_layer(W, b) for W, b in nnet_params)
    mu = linear_layer(W_mu, b_mu)
    log_sigmasq = linear_layer(W_sigma, b_sigma)

    def encode(X):
        h = nnet(X)
        return mu(h), log_sigmasq(h)

    return encode


def decoder(decoder_params):
    'a neural net with tanh layers until the final sigmoid layer'

    nnet_params, (W_out, b_out) = decoder_params[:-1], decoder_params[-1]
    nnet = compose(tanh_layer(W, b) for W, b in nnet_params)
    Y = sigmoid_layer(W_out, b_out)

    def decode(Z):
        return Y(nnet(Z))

    return decode


def get_zdim(decoder_params):
    return decoder_params[0][0].get_value().shape[0]


def make_objective(encoder_params, decoder_params):
    encode = encoder(encoder_params)
    decode = decoder(decoder_params)
    z_dim = get_zdim(decoder_params)

    def vlb(X, M, L):
        def sample_z(mu, log_sigmasq):
            eps = srng.normal((M, z_dim), dtype=theano.config.floatX)
            return mu + T.exp(0.5 * log_sigmasq) * eps

        def score_sample(Y):
            return -T.nnet.binary_crossentropy(Y, X).sum()

        mu, log_sigmasq = encode(X)
        kl_to_prior = 0.5 * T.sum(1. + log_sigmasq - mu**2. - T.exp(log_sigmasq))
        logpxz = sum(score_sample(decode(sample_z(mu, log_sigmasq)))
                     for l in xrange(L)) / floatX(L)

        return kl_to_prior + logpxz

    return vlb
