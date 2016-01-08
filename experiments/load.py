from __future__ import division
import string
import numpy as np
import theano

from util import floatX


def load_mice(N, permute=True, addnoise=True):
    data = np.load('data/images_for_vae.npy').astype(theano.config.floatX)
    if permute:
        data = np.random.permutation(data)
    data = data.reshape(data.shape[0], -1)[:N]
    data /= data.max()
    if addnoise:
        data += 1e-3*np.random.normal(size=data.shape)
    return theano.shared(floatX(data), borrow=True)


def load_mnist(N):
    raise NotImplementedError


def load_letters(which_letter=None):
    datadict = np.load('data/letters.npz')
    images, labels = datadict['images'], datadict['labels']

    if which_letter is not None:
        images = images[labels == string.lowercase.index(which_letter)]

    return theano.shared(floatX(images)), labels
