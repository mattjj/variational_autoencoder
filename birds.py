from __future__ import division
import numpy as np
import h5py
import matplotlib.pyplot as plt


def load_birds(file='data/birds/lhp33_2012-07-24/stft_features.mat'):
    f = h5py.File(file)
    mat = np.asarray(f['stft']['mat'])
    return mat['real'] + mat['imag']*1j


def plot(mat):
    db = 20. * np.log10(np.abs(mat))
    plt.matshow(db[0].T, clim=(-30,20))
    plt.savefig('birds.png')
    plt.close()
