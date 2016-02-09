from __future__ import division
import numpy as np
import numpy.random as npr
from scipy.signal import medfilt
import matplotlib.pyplot as plt
import cPickle as pickle
import gzip
import logging
import logging.config
logging.config.fileConfig('logging.conf')
from time import time

from vae.vae import make_gaussian_fitter
from vae.vae import encoder, gaussian_decoder, get_zdim, natural_to_mean  # temporary
from vae.optimization import sgd, adagrad, rmsprop, adadelta, adam, \
    momentum_sgd, nesterov
from vae.util import get_ndarrays
from vae.viz import plot_sample_grid

from load import load_mice

def forward_sampler(decoder_params):
    def sample(n):
        zdim = get_zdim(decoder_params)
        decode = gaussian_decoder(decoder_params)
        vals = decode(npr.randn(n, zdim))
        return vals[0].eval() if isinstance(vals, tuple) else vals.eval()
    return sample


def conditional_sampler(encoder_params, decoder_params, data):
    N = int(data.shape[0].eval())
    def sample(n):
        encode = encoder(encoder_params)
        decode = gaussian_decoder(decoder_params)
        subsampled_data = data[np.random.choice(N, size=n, replace=False)]
        vals = decode(natural_to_mean(encode(subsampled_data))[0])
        sampled_imagevecs = vals[0].eval() if isinstance(vals, tuple) else vals.eval()
        return np.hstack((sampled_imagevecs, subsampled_data.eval()))
    return sample


def plot(vals):
    tic = time()

    sample_forward = forward_sampler(decoder_params)
    plot_sample_grid(6, (30, 30), sample_forward)
    plt.savefig('mice_forward.png')
    plt.close()

    sample_conditional = conditional_sampler(encoder_params, decoder_params, trX)
    plot_sample_grid(6, (2*30, 30), sample_conditional)
    plt.savefig('mice_conditional.png')
    plt.close()

    plt.plot(medfilt(vals, 101)[:-50])
    plt.savefig('training_progress_this_epoch.png')
    plt.close()

    print 'plotting took {} sec'.format(time() - tic)

if __name__ == '__main__':
    logging.info('\n\nStarting experiment!')
    np.random.seed(0)

    N = 750000  # 750k is about the memory limit on 3GB GPU
    # trX = load_mice(N, 'data/sod1-shrunk.npy')
    trX = load_mice(N, 'data/sod1-newest.npy')

    encoder_params, decoder_params, fit = \
        make_gaussian_fitter(trX, 10, [200, 200], [200, 200], callback=plot)

    fit(1, 50, 1, adadelta())
    fit(1, 250, 1, adadelta())
    fit(1, 250, 1, rmsprop(1e-4))
    # fit(25, 250, 1, rmsprop(1e-6))
    # fit(25, 2000, 1, rmsprop(1e-7))

    params = get_ndarrays(encoder_params), get_ndarrays(decoder_params)
    with gzip.open('temp.pkl.gz', 'w') as f:
        pickle.dump(params, f, protocol=-1)
