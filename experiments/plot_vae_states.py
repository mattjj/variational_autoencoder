from __future__ import division
import numpy as np
import numpy.random as npr
import gzip
import cPickle as pickle
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from vae.viz import numpy_encoder as encoder

partial_flatten = lambda x: np.reshape(x, (x.shape[0], -1))

def load_data(filename):
    if filename.endswith('.npy'):
        mices = np.load(filename)
    else:
        with gzip.open(filename, 'r') as infile:
            datadict = pickle.load(infile)
        mices = np.concatenate(datadict.values())
    # TODO normalize each instead of normalize all
    mices -= mices.min()
    mices /= mices.max()
    return partial_flatten(mices)

def load_params(filename):
    with gzip.open(filename, 'r') as infile:
        tup = pickle.load(infile)
    return tup

def cov(X):
    X = X - X.mean(0)
    return np.dot(X.T, X) / X.shape[0]

def autoregression_pairplot(
        data, color='b', cmap=plt.cm.Blues, affine=True, kwargs1={}, kwargs2={}):
    def regression_matrix(y, x):
        xxT, yxT = np.dot(x.T, x), np.dot(y.T, x)
        return np.linalg.solve(xxT, yxT.T).T

    def partial_residuals(y, x, A=None):
        A = A if A is not None else regression_matrix(y, x)
        residuals = y - np.dot(x, A.T)
        return residuals[...,None] + x[:,None,:] * A[None,...]

    x, y = data[:-1], data[1:]

    if affine:
        x = np.hstack((x, np.ones((x.shape[0], 1))))

    A = regression_matrix(y, x)
    r = partial_residuals(y, x, A)

    fig = plt.figure(figsize=(12, 12))
    gs = GridSpec(*A.shape)
    for (i, j), aij in np.ndenumerate(A):
        plt.subplot(gs[i,j])
        # plt.plot(x[:,j], r[:,i,j], '.', color=color, **kwargs1)
        plt.hexbin(x[:,j], r[:,i,j], bins='log', cmap=cmap, **kwargs1)
        t = np.linspace(np.min(x[:,j]), np.max(x[:,j]), 100)
        plt.plot(t, aij * t, '-', color=color, **kwargs2)
        plt.xticks([])
        plt.yticks([])

def make_plotter(encode, mices):
    def get_means(x):
        J, h = encode(x)
        return h / (-2.*J)

    stds = np.std(get_means(mices[:20000]), axis=0)
    U, _, _ = np.linalg.svd(cov(mices))

    def plot_means(start, stop, scaled=False, figsize=(12,8)):
        means = get_means(mices[start:stop])
        D = means.shape[1]
        fig, axs = plt.subplots(D+1, 1, figsize=figsize, sharex=True, sharey=True)
        axs[0].plot(means / stds if scaled else means)
        for mean, ax in zip(means.T, axs[1:]):
            ax.plot(mean)
        plt.savefig('vae_means.png')
        plt.close()

    def plot_means_flat(start, stop, scaled=False, **kwargs):
        means = get_means(mices[start:stop])
        plt.figure(figsize=(6, 6))
        plt.plot(means[:,0], means[:,1], 'b.', **kwargs)
        plt.savefig('vae_means_flat.png')
        plt.close()

    def plot_pca(start, stop, **kwargs):
        vals = np.dot(mices[start:stop], U[:,:10])
        plt.figure(figsize=(6, 6))
        plt.plot(vals[:,0], vals[:,1], 'r.', **kwargs)
        plt.savefig('pca_mices.png')
        plt.close()

    def pairplot(start, stop, which='pca', **kwargs):
        if which == 'pca':
            data = np.dot(mices[start:stop], U[:,:10])
        else:
            data = get_means(mices[start:stop])

        autoregression_pairplot(data, **kwargs)
        plt.savefig('autoregression_{}_pairplot.png'.format(which))
        plt.close()

    return plot_means, plot_means_flat, plot_pca, pairplot

if __name__ == "__main__":
    # mices = load_data('data/new-dawn-corrected-shrunk.npy')
    mices = load_data('data/new-dawn-corrected-shrunk3.pkl.gz')[:200000:2]
    # encoder_params, _ = load_params('vae_params.pkl.gz')
    # encoder_params, _ = load_params('best_init.pkl.gz')
    # _, _, encoder_params = load_params('../svae/lds_svae_params_vaeinit.pkl.gz')
    _, _, encoder_params = load_params('../svae/lds_svae_params_noinit.pkl.gz')

    encode = encoder(encoder_params, 7.)
    plot, plot_flat, plot_pca, pairplot = make_plotter(encode, mices)

    plot(500, 1000)
    plot_flat(0, 20000, alpha=0.1)
    plot_pca(0, 20000, alpha=0.1)

    pairplot(0, 36000, which='vae', color='b', cmap=plt.cm.Blues)
    pairplot(0, 36000, which='pca', color='r', cmap=plt.cm.Reds)
