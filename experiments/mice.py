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

def forward_sampler(decoder_params, tanh_scale):
    def sample(n):
        zdim = get_zdim(decoder_params)
        decode = gaussian_decoder(decoder_params, tanh_scale)
        vals = decode(npr.randn(n, zdim))
        return vals[0].eval() if isinstance(vals, tuple) else vals.eval()
    return sample


def conditional_sampler(encoder_params, decoder_params, tanh_scale, data):
    N = int(data.shape[0].eval())
    def sample(n):
        encode = encoder(encoder_params, tanh_scale)
        decode = gaussian_decoder(decoder_params, tanh_scale)
        subsampled_data = data[np.random.choice(N, size=n, replace=False)]
        vals = decode(natural_to_mean(encode(subsampled_data))[0])
        sampled_imagevecs = vals[0].eval() if isinstance(vals, tuple) else vals.eval()
        return np.hstack((sampled_imagevecs, subsampled_data.eval()))
    return sample


def plot_training_progress(ys):
    def cumulative_ranges(lens):
        stops = np.cumsum(lens)
        return [range(stop - l, stop) for stop, l in zip(stops, lens)]

    xs = cumulative_ranges(map(len, ys))
    y = np.concatenate(ys)

    fig, (ax1, ax2) = plt.subplots(2, 1)
    lines = [ax1.plot(x, y) for x, y in zip(xs, ys)]
    ax1.set_ylim(np.min(y), np.percentile(ys[0], 90.))

    ax2.plot(xs[-1], ys[-1], lines[-1][-1].get_color())
    ax2.plot(xs[-1][12:-12], medfilt(ys[-1], 25)[12:-12], 'k')
    ax2.set_ylim(np.min(ys[-1]), np.percentile(ys[-1], 99.))


saved = lambda: None
saved.epoch = 0
saved.valss = []
def plot(vals):
    tic = time()

    sample_forward = forward_sampler(decoder_params, tanh_scale)
    plot_sample_grid(8, (30, 30), sample_forward)
    plt.savefig('mice_forward.png')
    plt.close()

    sample_conditional = conditional_sampler(encoder_params, decoder_params, tanh_scale, trX)
    plot_sample_grid(6, (2*30, 30), sample_conditional)
    plt.savefig('mice_conditional.png')
    plt.close()

    saved.valss.append(vals)
    plot_training_progress(saved.valss)
    plt.savefig('training_progress.png'.format(saved.epoch))
    plt.close()
    saved.epoch += 1

    print 'plotting took {} sec'.format(time() - tic)

def save(encoder_params, decoder_params):
    params = get_ndarrays(encoder_params), get_ndarrays(decoder_params)
    with gzip.open('vae_params.pkl.gz', 'w') as f:
        pickle.dump(params, f, protocol=-1)
        print 'saved!'

if __name__ == '__main__':
    logging.info('\n\nStarting experiment!')
    np.random.seed(0)

    N = 750000  # 750k is about the memory limit on 3GB GPU
    # trX = load_mice(N, 'data/sod1-shrunk.npy')
    # trX = load_mice(N, 'data/sod1-newest.npy')
    # trX = load_mice(N, 'data/new-dawn-corrected-shrunk.npy')
    # trX = load_mice(N, 'data/new-dawn-corrected-shrunk2.pkl.gz')
    trX = load_mice(N, 'data/new-dawn-corrected-shrunk3.pkl.gz')
    tanh_scale = 7.

    encoder_params, decoder_params, fit = make_gaussian_fitter(
            trX, 10, [200, 200], [200, 200], tanh_scale=tanh_scale, callback=plot)

    fit(1, 50, 1, adadelta())
    fit(1, 250, 1, adadelta())
    fit(15, 250, 1, adam(1e-3))
    save(encoder_params, decoder_params)
    fit(25, 500, 1, adam(5e-4))
    save(encoder_params, decoder_params)
    fit(50, 1000, 1, adam(5e-5))
    save(encoder_params, decoder_params)
    fit(50, 1000, 1, adam(1e-5))
    save(encoder_params, decoder_params)
    fit(250, 2000, 1, adam(5e-6))
    save(encoder_params, decoder_params)

    # fit(10, 250, 1, rmsprop(1e-4))
    # fit(1, 250, 1, rmsprop(1e-5))
    # fit(1, 250, 1, rmsprop(1e-6))

    # TODO try single hidden layer, probably much easier to train
    # actually not any easier to train? taking comparably long, forward
    # generated mice look a bit funny (though conditional ones look great)
    # maybe it was the dataset? try switching back to 2 (from 3)
