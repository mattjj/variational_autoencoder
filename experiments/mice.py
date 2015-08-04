import numpy as np
import logging
import logging.config
logging.config.fileConfig('logging.conf')
import matplotlib.pyplot as plt

from util import flatten
from load import load_mice
from vae import make_gaussian_fitter
from optimization import sgd, adagrad, rmsprop, adadelta, adam, \
    momentum_sgd, nesterov
from viz import sample_grid

all_vals = []
def monitor_progress(epoch_vals):
    global decoder_params, all_vals

    def print_first_decoder_weights_spectrum():
        def get_first_decoder_weights(decoder_params):
            return decoder_params[0][0].get_value()
        s = np.linalg.svd(get_first_decoder_weights(decoder_params))[1]
        print s[np.argsort(-s)]

    print_first_decoder_weights_spectrum()

    all_vals.append(map(float,epoch_vals))
    plt.plot(flatten(all_vals))
    plt.savefig('costvals.png')
    plt.close(plt.gcf().number)

    sample_grid(20, decoder_params)
    plt.savefig('samples_{}.png'.format(len(all_vals)))
    plt.close(plt.gcf().number)

# TODO plot as a function of time, add legend for params during sweep
# TODO first tried 1e-6 stepsize, first sing. val growing while second is still
# 0.1768, try bigger stepsize (like 1e-4)
# TODO save plots in an experiment directory, which also saves the code of this
# file when it's run
# TODO should i be subtracting off the mean or something?
# TODO sshfs is slow... run a webserver or something? or tail a log file and
# plot locally? probably web server, less bespoke. maybe sshfs to jefferson and
# run a webserver there.

# it seems to be learning to draw really slowly, especially the background
# color, which it should be learning is always zero...
# the singular values are really shit with this slow training, too


if __name__ == '__main__':
    logging.info('\n\nStarting experiment!')

    N = 500000  # 750k is about the memory limit on 3GB GPU
    trX = load_mice(N)

    np.random.seed(2)
    encoder_params, decoder_params, fit = \
        make_gaussian_fitter(trX, 20, [500], [500], monitor_progress)

    fit(1, 50, 1, adam(1e-5))
    fit(3, 250, 1, adam(1e-5))
    fit(9, 1250, 1, adam(1e-5))

    fit(27, 5000, 1, adam(1e-5))

