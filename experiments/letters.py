from __future__ import division
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import cPickle as pickle
import string
import gzip
import theano

from vae.vae import make_binary_fitter, binary_decoder
from vae.optimization import adadelta, rmsprop
from vae.util import get_ndarrays, floatX
from vae.viz import plot_sample_grid

from load import load_letters


if __name__ == "__main__":
    npr.seed(0)

    trX, labels = load_letters('e')

    encoder_params, decoder_params, fit = make_binary_fitter(trX, 5, [200], [200])

    fit(1, 50, 1, adadelta())
    fit(1, 250, 1, adadelta())
    fit(400, 50, 1, rmsprop(1e-3))
    fit(200, 500, 10, rmsprop(1e-4))

    params = get_ndarrays(encoder_params), get_ndarrays(decoder_params)
    with gzip.open('letter_params.pkl.gz', 'w') as f:
        pickle.dump(params, f, protocol=-1)


    plot_sample_grid(5, decoder_params, (16, 8), binary_decoder)
    plt.show()
