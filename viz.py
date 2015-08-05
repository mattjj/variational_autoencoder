from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

from util import sigmoid
from vae import encoder, gaussian_decoder, get_zdim


def make_grid(grid_sidelen, imagevecs):
    im_sidelen = int(np.sqrt(imagevecs.shape[1]))
    reshaped = imagevecs.reshape(grid_sidelen,grid_sidelen,im_sidelen,im_sidelen)
    return np.vstack([np.hstack([
        img.reshape(im_sidelen,im_sidelen)
        for img in col]) for col in reshaped])


def generate_samples(n, decoder_params):
    zdim = get_zdim(decoder_params)
    decode = gaussian_decoder(decoder_params)
    return decode(np.random.randn(n, zdim))[0].eval()


def sample_grid(sidelen, decoder_params):
    imagevecs = generate_samples(sidelen**2, decoder_params)
    return make_grid(sidelen, imagevecs)


def training_grid(sidelen, trX):
    imagevecs = np.random.permutation(trX.get_value())[:sidelen**2]
    return make_grid(sidelen, imagevecs)


def encode_seq(X, encoder_params):
    encode = encoder(encoder_params)
    return encode(X)[0].eval()
