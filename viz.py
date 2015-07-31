from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

from util import sigmoid


def decode(z, W4, W5, b4, b5):
    W4, W5, b4, b5 = [_.get_value() for _ in [W4, W5, b4, b5]]
    h_decoder = np.tanh(z.dot(W4) + b4)
    return sigmoid(h_decoder.dot(W5) + b5)


def generate_samples(n, W4, W5, b4, b5):
    z_dim = W4.get_value().shape[0]
    return decode(np.random.randn(n, z_dim), W4, W5, b4, b5)


def make_grid(grid_sidelen, imagevecs):
    im_sidelen = int(np.sqrt(imagevecs.shape[1]))
    reshaped = imagevecs.reshape(grid_sidelen,grid_sidelen,im_sidelen,im_sidelen)
    return np.vstack([np.hstack([
        img.reshape(im_sidelen,im_sidelen)
        for img in col]) for col in reshaped])


def sample_grid(grid_sidelen, W4, W5, b4, b5):
    imagevecs = generate_samples(grid_sidelen**2, W4, W5, b4, b5)
    plt.matshow(make_grid(grid_sidelen, imagevecs))


def encode(x, W1, W2, b1, b2):
    # returns the mean of the corresponding variational factor
    h_encoder = np.tanh(x.dot(W1) + b1)
    return h_encoder.dot(W2) + b2


# what else did people do for mnist visualization?
