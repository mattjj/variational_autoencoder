from __future__ import division
import numpy as np
import numpy.random as npr
import cPickle as pickle
import string
import gzip
import theano

from vae.vae import make_binary_fitter
from vae.optimization import adadelta
from vae.util import get_ndarrays, floatX

from load import load_letters


if __name__ == "__main__":
    npr.seed(0)

    images, labels = load_letters()
    trX = theano.shared(floatX(images[labels == string.lowercase.index('g')]))

    encoder_params, decoder_params, fit = make_binary_fitter(trX, 3, [200], [200])

    fit(1, 50, 1, adadelta())
    fit(1, 250, 1, adadelta())

    params = get_ndarrays(encoder_params), get_ndarrays(decoder_params)
    with gzip.open('letter_params.pkl.gz', 'w') as f:
        pickle.dump(params, f, protocol=-1)
