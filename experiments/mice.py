import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
import gzip
import logging
import logging.config
logging.config.fileConfig('logging.conf')
from scipy.signal import medfilt

from vae.vae import make_gaussian_fitter, gaussian_decoder
from vae.optimization import sgd, adagrad, rmsprop, adadelta, adam, \
    momentum_sgd, nesterov
from vae.util import get_ndarrays
from vae.viz import plot_sample_grid

from load import load_mice


def plot(vals):
    plot_sample_grid(6, decoder_params, (30, 30), gaussian_decoder)
    plt.savefig('mice.png')
    plt.close()

    plt.plot(medfilt(vals, 101)[:-50])
    plt.savefig('training_progress_this_epoch.png')
    plt.close()

if __name__ == '__main__':
    logging.info('\n\nStarting experiment!')
    np.random.seed(0)

    N = 750000  # 750k is about the memory limit on 3GB GPU
    # trX = load_mice(N, 'data/sod1-shrunk.npy')
    trX = load_mice(N, 'data/sod1-newest.npy')

    encoder_params, decoder_params, fit = \
        make_gaussian_fitter(trX, 5, [200, 200], [200, 200], callback=plot)

    fit(1, 50, 1, adadelta())
    fit(10, 50, 1, adadelta())
    fit(1, 250, 1, rmsprop(1e-4))
    fit(25, 250, 1, rmsprop(1e-6))
    fit(25, 2000, 1, rmsprop(1e-7))

    params = get_ndarrays(encoder_params), get_ndarrays(decoder_params)
    with gzip.open('mice_k2_params_tanh10.pkl.gz', 'w') as f:
        pickle.dump(params, f, protocol=-1)
