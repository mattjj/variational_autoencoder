from __future__ import division
import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from util import floatX, flatten
from nnet import compose, tanh_layer, sigmoid_layer, linear_layer, init_layer

srng = RandomStreams(seed=1)


##########
#  util  #
##########

def get_zdim(decoder_params):
    return decoder_params[0][0].get_value().shape[0]


####################
#  initialization  #
####################

def init_encoder(N_in, hdims, N_out):
    dims = [N_in] + hdims
    nnet_params = [init_layer(shape) for shape in zip(dims[:-1], dims[1:])]
    W_mu, b_mu = init_layer((hdims[-1], N_out))
    W_sigma, b_sigma = init_layer((hdims[-1], N_out))
    return nnet_params + [(W_mu, b_mu), (W_sigma, b_sigma)]


def init_binary_decoder(N_in, hdims, N_out):
    dims = [N_in] + hdims + [N_out]
    return [init_layer(shape) for shape in zip(dims[:-1], dims[1:])]


init_gaussian_decoder = init_encoder


def _make_initializer(init_decoder):
    def init_params(Nx, Nz, encoder_hdims, decoder_hdims):
        encoder_params = init_encoder(Nx, encoder_hdims, Nz)
        decoder_params = init_decoder(Nz, decoder_hdims, Nx)
        return encoder_params, decoder_params, flatten((encoder_params, decoder_params))
    return init_params


init_binary_params = _make_initializer(init_binary_decoder)
init_gaussian_params = _make_initializer(init_gaussian_decoder)


##########################
#  enoders and decoders  #
##########################

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


def gaussian_decoder(decoder_params):
    'just like the (gaussian) encoder but means are mapped through a logistic'

    code = encoder(decoder_params)

    def decode(Z):
        mu, log_sigmasq = code(Z)
        return T.nnet.sigmoid(mu), log_sigmasq

    return decode


def binary_decoder(decoder_params):
    'a neural net with tanh layers until the final sigmoid layer'

    nnet_params, (W_out, b_out) = decoder_params[:-1], decoder_params[-1]
    nnet = compose(tanh_layer(W, b) for W, b in nnet_params)
    Y = sigmoid_layer(W_out, b_out)

    def decode(Z):
        return Y(nnet(Z))

    return decode


#########################
#  objective functions  #
#########################

def binary_loglike(X, Y):
    return -T.nnet.binary_crossentropy(Y, X).sum()


def gaussian_loglike(X, params):
    mu, log_sigmasq = params
    return -0.5*T.sum(
        (np.log(2.*np.pi) + log_sigmasq) + (X - mu)**2. / T.exp(log_sigmasq))


def _make_objective(decoder, loglike):
    def make_objective(encoder_params, decoder_params):
        encode = encoder(encoder_params)
        decode = decoder(decoder_params)
        z_dim = get_zdim(decoder_params)

        def vlb(X, N, M, L):
            def sample_z(mu, log_sigmasq):
                eps = srng.normal((M, z_dim), dtype=theano.config.floatX)
                return mu + T.exp(0.5 * log_sigmasq) * eps

            def kl_to_prior(mu, log_sigmasq):
                return -0.5*T.sum(1. + log_sigmasq - mu**2. - T.exp(log_sigmasq))

            mu, log_sigmasq = encode(X)
            logpxz = sum(loglike(X, decode(sample_z(mu, log_sigmasq)))
                        for l in xrange(L)) / floatX(L)

            return N/M * (-kl_to_prior(mu, log_sigmasq) + logpxz)

        return vlb
    return make_objective


make_gaussian_objective = _make_objective(gaussian_decoder, gaussian_loglike)
make_binary_objective = _make_objective(binary_decoder, binary_loglike)
