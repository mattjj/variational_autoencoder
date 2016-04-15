from __future__ import division
import numpy as np
import string
import operator as op
import cPickle as pickle
import gzip
import theano

from scipy.ndimage.filters import gaussian_filter

from util import floatX

def load(filename, concatenate_dict=True):
    def standardize(d):
        recenter = lambda d: d - np.percentile(d, 0.01)
        rescale = lambda d: d / np.percentile(d, 99.99)
        return rescale(recenter(d))

    if filename.endswith('.npy'):
        return standardize(np.load(filename))
    else:
        openfile = open if filename.endswith('.pkl') else gzip.open
        with openfile(filename, 'r') as infile:
            datadict = pickle.load(infile)
        get_images = lambda v: v['images'] if isinstance(v, dict) else v
        datadict = {k:standardize(get_images(v)) for k, v in datadict.iteritems()
                    if k not in {'SOD1-LC-1-15-_04', 'SOD1-LC-1-15-_24'}}
        if concatenate_dict:
            data = map(op.itemgetter(1), sorted(datadict.items(), key=op.itemgetter(0)))
            return np.concatenate(datadict.values())
        return datadict


def load_mice(N, file='data/images_for_vae.npy', permute=True, addnoise=True):
    data = load(file).astype(theano.config.floatX)
    if permute:
        data = np.random.permutation(data)
    data = data.reshape(data.shape[0], -1)
    if addnoise:
        data += 1e-3*np.random.normal(size=data.shape)
    print 'loaded {} frames from {}'.format(min(N, len(data)), file)
    return theano.shared(floatX(data[:N]), borrow=True)


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
