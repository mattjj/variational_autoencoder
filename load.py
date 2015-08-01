from __future__ import division
import numpy as np
import theano

from util import floatX


def load_mice(N):
    data = np.load('data/images_for_vae.npy').astype(theano.config.floatX)
    data = np.random.permutation(data.reshape(data.shape[0], -1))[:N]
    data /= data.max()
    return theano.shared(floatX(data), borrow=True)


def load_mnist(N):
    raise NotImplementedError
