from __future__ import division
import string
import numpy as np
import cPickle as pickle
import theano

from scipy.ndimage.filters import gaussian_filter

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

    return theano.shared(floatX(images), borrow=True), labels


def load_pendulum(N, permute=True, addnoise=True):
    with open('data/pendulous.pkl') as infile:
        images = gaussian_filter(pickle.load(infile), 0.75).astype(theano.config.floatX)

    if permute:
        images = np.random.permutation(images)

    images = images[:N]

    images -= images.min()
    images /= images.max()

    if addnoise:
        images += 1e-2*np.random.normal(size=images.shape)

    images = np.reshape(images, (images.shape[0], -1))

    return theano.shared(floatX(images), borrow=True)
