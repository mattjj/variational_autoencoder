import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
import gzip
import logging
import logging.config
logging.config.fileConfig('logging.conf')

from vae.vae import make_gaussian_fitter, gaussian_decoder, encoder
from vae.optimization import sgd, adagrad, rmsprop, adadelta, adam, \
    momentum_sgd, nesterov
from vae.util import get_ndarrays
from vae.viz import plot_sample_grid

from load import load_mice


def plot():
    plot_sample_grid(6, decoder_params, (30, 30), gaussian_decoder)
    plt.savefig('mice.png')

if __name__ == '__main__':
    logging.info('\n\nStarting experiment!')
    np.random.seed(0)

    N = 750000  # 750k is about the memory limit on 3GB GPU
    # trX = load_mice(N, 'data/sod1-shrunk.npy')
    trX = load_mice(N, 'data/sod1-new-shrunk.npy')

    encoder_params, decoder_params, fit = \
        make_gaussian_fitter(trX, 10, [200], [200])

    fit(1, 50, 1, adadelta())
    plot()
    fit(1, 250, 1, adadelta())
    plot()
    fit(10, 250, 1, rmsprop(1e-4))
    plot()

    fit(25, 250, 1, rmsprop(1e-5))
    plot()

    params = get_ndarrays(encoder_params), get_ndarrays(decoder_params)
    with gzip.open('mice_k2_params.pkl.gz', 'w') as f:
        pickle.dump(params, f, protocol=-1)

    # # making a reconstructed dataset
    # from vae.vae import natural_to_mean
    # X = np.load('data/sod1-shrunk.npy')
    # X = X.reshape(X.shape[0], -1)
    # X /= X.max()

    # encode = encoder(encoder_params)
    # decode = gaussian_decoder(decoder_params)
    # def reconstruct(x):
    #     return decode(natural_to_mean(encode(x))[0])[0].eval()
    # Xtilde = np.vstack(map(reconstruct, np.array_split(X, 1000)))
    # np.save('data/sod1-reconstructed.npy', Xtilde)
    # print 'done!'

    # # comparing versions
    # from vae.viz import make_grid
    # def compare(X, Y, sidelen=10, seed=0):
    #     idx = np.random.RandomState(seed).choice(X.shape[0], size=sidelen**2, replace=False)

    #     def plot(Z, filename):
    #         plt.matshow(make_grid(sidelen, Z[idx], (30, 30)))
    #         plt.savefig(filename)
    #         plt.close()

    #     plot(X, 'orig.png')
    #     plot(Y, 'reconstructed.png')
