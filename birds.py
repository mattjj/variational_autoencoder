from __future__ import division
import numpy as np
import numpy.random as npr
import h5py
import matplotlib.pyplot as plt


def load_birds(file='data/birds/lhp33_2012-07-24/stft_features.mat', addnoise=True, permute=True):
    f = h5py.File(file)
    mat = np.asarray(f['stft']['mat'])
    cmat = mat['real'] + mat['imag']*1j

    db = 20. * np.log10(np.abs(cmat))

    clipped = np.clip(db, -30, 20)
    clipped -= clipped.min()
    clipped /= clipped.max()

    if addnoise:
        clipped += 1e-3 * npr.randn(*clipped.shape)

    # trials x time x freqs
    return np.reshape(clipped, (-1, clipped.shape[-1]))


def plot(db):
    plt.matshow(db[:1000].T)
    plt.savefig('birds.png')
    plt.close()
