from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import cm

from vae import encoder, gaussian_decoder, get_zdim, unpack_gaussian_params
from nnet import compose, numpy_tanh_layer, numpy_linear_layer
from util import sigmoid, reshape_square


def make_grid(grid_sidelen, imagevecs):
    im_sidelen = int(np.sqrt(imagevecs.shape[1]))
    shape = 2*(grid_sidelen,) + 2*(im_sidelen,)
    reshaped = imagevecs.reshape(shape)

    return np.vstack(
        [np.hstack([reshape_square(img) for img in col]) for col in reshaped])


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


class Interactive(object):
    def __init__(self, draw_func, init_image):
        self.fig, (self.ax, self.imax) = fig, (ax, imax) = \
            plt.subplots(1, 2, figsize=(10, 5), facecolor='white')
        self.canvas = canvas = fig.canvas

        self.draw_func = draw_func

        self._init_controlaxis(ax)
        self._init_imageaxis(imax, init_image)
        fig.tight_layout()

        canvas.mpl_connect('draw_event', self.draw)
        canvas.mpl_connect('button_press_event', self.button_press)
        canvas.mpl_connect('button_release_event', self.button_release)
        canvas.mpl_connect('motion_notify_event', self.motion_notify)

    ### initialization

    def _init_controlaxis(self, ax):
        ax.axis([-1,1,-1,1])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.autoscale(False)
        self.circle = circle = Circle(
            (0,0), radius=0.02, facecolor='r', animated=True)
        ax.add_patch(circle)
        self._dragging = False

    def _init_imageaxis(self, imax, init_image):
        imax.set_axis_off()
        self.image = imax.imshow(init_image, cmap=cm.YlGnBu)
        imax.autoscale(False)

    ### matplotlib callbacks

    def draw(self, event):
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.ax.draw_artist(self.circle)
        self.imax.draw_artist(self.image)
        self.canvas.blit(self.ax.bbox)
        self.canvas.blit(self.imax.bbox)

    def button_press(self, event):
        if event.inaxes is self.ax and event.button == 1:
            self._dragging = True
            self._update(event.xdata, event.ydata)

    def button_release(self, event):
        if event.button == 1:
            self._dragging = False

    def motion_notify(self, event):
        if event.inaxes is self.ax and event.button == 1:
            if self._dragging:
                self._update(event.xdata, event.ydata)

    ### updating

    def _update(self, x, y):
        self._reposition_circle(x, y)
        self._update_image(x, y)

    def _reposition_circle(self, x, y):
        self.circle.center = x, y
        self.canvas.restore_region(self.background)
        self.ax.draw_artist(self.circle)
        self.canvas.blit(self.ax.bbox)

    def _update_image(self, x, y):
        self.image.set_data(self.draw_func(x, y))
        self.imax.draw_artist(self.image)
        self.canvas.blit(self.imax.bbox)


def numpy_gaussian_decoder(decoder_params):
    # mostly redundant code with encoder and gaussian_decoder in vae.py
    nnet_params, (W_mu, b_mu), _ = \
        unpack_gaussian_params(decoder_params)
    nnet = compose(numpy_tanh_layer(W, b) for W, b in nnet_params)
    mu = numpy_linear_layer(W_mu, b_mu)

    def decode(X):
        return sigmoid(mu(nnet(X)))

    return decode


def run_interactive(decoder_params):
    zdim = get_zdim(decoder_params)
    decode = numpy_gaussian_decoder(decoder_params)
    vec = np.zeros(zdim)

    def draw(x, y):
        vec[:2] = (x,y)
        return reshape_square(decode(vec))

    return Interactive(draw, draw(0,0))
