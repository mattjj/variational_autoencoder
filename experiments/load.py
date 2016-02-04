from __future__ import division
import string
import numpy as np
import cPickle as pickle
import theano
import h5py

from scipy.ndimage.filters import gaussian_filter

from util import floatX


def load_mice(N, file='data/images_for_vae.npy', chunksize=None, addnoise=True):
    data = np.load(file).astype(theano.config.floatX)
    if chunksize is None:
        data = np.random.permutation(data)
    else:
        num_chunks = data.shape[0] // chunksize
        data = data[:chunksize*num_chunks]
        data = np.concatenate(npr.permutation(np.split(data, num_chunks))[:N // chunksize])

    data = data.reshape(data.shape[0], -1)[:N]
    data -= data.min()
    data /= data.max()
    if addnoise:
        data += 1e-3*np.random.normal(size=data.shape)
    print 'loaded %d frames from %s' % (data.shape[0], file)
    return theano.shared(floatX(data), borrow=True)

def load_mice_k2(N, file='data/sod1-shrunk.npy', *args, **kwargs):
    return load_mice(N, file, *args, **kwargs)


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
