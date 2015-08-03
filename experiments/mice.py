import numpy as np
import logging.config
logging.config.fileConfig('logging.conf')

from load import load_mice
from vae import make_gaussian_fitter
from optimization import sgd, adagrad, rmsprop, adadelta, adam, \
    momentum_sgd, nesterov


if __name__ == '__main__':
    np.random.seed(2)

    N = 750000  # 750k is about the memory limit on 3GB GPU
    trX = load_mice(N)

    encoder_params, decoder_params, fit = \
        make_gaussian_fitter(trX, 20, [50], [50])

    fit(1,  20, 1, adam(1e-6))
    fit(3,  20, 1, adam(1e-5))
    fit(9,  20, 1, adam(1e-5))
    fit(27, 20, 1, adam(1e-5))
    fit(1000, 5000, 1, adam(1e-6))

    # TODO call viz code as we go
    # TODO add plotting of training curves
