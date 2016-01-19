from __future__ import division
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import cPickle as pickle
import string
import gzip
import theano

from vae.vae import make_gaussian_fitter, gaussian_decoder
from vae.optimization import adadelta, rmsprop, adam, nesterov
from vae.util import get_ndarrays, floatX
from vae.viz import plot_sample_grid

from load import load_pendulum


def plot():
    plot_sample_grid(5, decoder_params, (21, 21), gaussian_decoder)
    plt.savefig('pendulum.png')

if __name__ == "__main__":
    npr.seed(0)

    trX = load_pendulum(100)

    encoder_params, decoder_params, fit = \
        make_gaussian_fitter(trX, 3, [200], [200])  # 2 also works well, 1 not quite as well

    fit(1*500, 50, 1, adadelta())
    plot()
    fit(25*500, 50, 1, adadelta())
    plot()
    fit(50*500, 100, 1, rmsprop(1e-4))
    plot()
    fit(50*500, 100, 1, rmsprop(1e-5))
    plot()
    fit(50*500, 100, 1, rmsprop(1e-6))
    plot()

    params = get_ndarrays(encoder_params), get_ndarrays(decoder_params)
    with gzip.open('pendulum_params.pkl.gz', 'w') as f:
        pickle.dump(params, f, protocol=-1)

    plt.show()

    # TODO try adam
