import numpy as np
import logging
import logging.config
logging.config.fileConfig('logging.conf')
import matplotlib.pyplot as plt

from vae.util import flatten
from vae.vae import make_gaussian_fitter
from vae.optimization import sgd, adagrad, rmsprop, adadelta, adam, \
    momentum_sgd, nesterov
from vae.viz import sample_grid

from load import load_mice


def monitor_progress(epoch_vals, all_vals=[]):
    global decoder_params

    def print_first_decoder_weights_spectrum():
        def get_first_decoder_weights(decoder_params):
            return decoder_params[0][0].get_value()

        s = np.linalg.svd(get_first_decoder_weights(decoder_params))[1]
        print s[np.argsort(-s)]

    def plot_costvals():
        all_vals.append(map(float, epoch_vals))
        plt.plot(flatten(all_vals))
        plt.savefig('figures/costvals.png')
        plt.close(plt.gcf().number)
        print_first_decoder_weights_spectrum()

    def show_sampled_images():
        plt.matshow(sample_grid(20, decoder_params))
        plt.savefig('figures/samples_{}.png'.format(len(all_vals)))
        plt.close(plt.gcf().number)

    print_first_decoder_weights_spectrum()
    show_sampled_images()
    plot_costvals()
    print 'saved files for epoch {}!'.format(len(all_vals))


if __name__ == '__main__':
    logging.info('\n\nStarting experiment!')
    np.random.seed(0)

    N = 750000  # 750k is about the memory limit on 3GB GPU
    trX = load_mice(N)

    encoder_params, decoder_params, fit = \
        make_gaussian_fitter(trX, 20, [200, 200], [200, 200], monitor_progress)

    fit(1, 50, 1, adadelta())
    fit(1, 250, 1, adadelta())
    fit(10, 500, 1, rmsprop(1e-4))
    fit(25, 500, 1, rmsprop(1e-5))
    fit(25, 1000, 1, rmsprop(1e-5))
