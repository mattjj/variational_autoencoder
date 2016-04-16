from __future__ import division
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import cm

from vae import encoder, gaussian_decoder, get_zdim, unpack_gaussian_params
from nnet import compose, numpy_tanh_layer, numpy_linear_layer
from util import sigmoid, reshape_square


def make_grid(grid_sidelen, imagevecs, imshape):
    shape = 2*(grid_sidelen,) + imshape
    reshaped = imagevecs.reshape(shape)

    return np.vstack(
        [np.hstack([np.reshape(img, imshape) for img in col]) for col in reshaped])


def training_grid(sidelen, trX, imshape, seed=None):
    rng = npr if seed is None else npr.RandomState(seed=seed)
    imagevecs = rng.permutation(trX.get_value())[:sidelen**2]
    return make_grid(sidelen, imagevecs, imshape)


def plot_sample_grid(sidelen, imshape, samplefn):
    grid = make_grid(sidelen, samplefn(sidelen**2), imshape)
    show_sample_matrix(grid, sidelen, imshape)


def show_sample_matrix(grid, sidelen, imshape, cmap='gray', outfile=None):
    plt.matshow(grid)
    ax = plt.gca()
    xx, yy = imshape
    ax.set_yticks(np.arange(0, (sidelen+1)*xx, xx) - 0.5)
    ax.set_xticks(np.arange(0, (sidelen+1)*yy, yy) - 0.5)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    plt.set_cmap(cmap)
    # plt.axis('off')
    plt.gca().set_frame_on(False)
    plt.grid(True, color='r', linestyle='-')

    if outfile:
        plt.savefig(outfile, transparent=True, dpi=200, bbox_inches='tight')
        plt.close()


def regular_grid(sidelen, decoder_params, imshape, limits=[-2,2,-2,2], axes=None,
                 corners=None, rand_scale=1., seed=None, decoder=None):
    if decoder is None:
        from vae import gaussian_decoder as decoder
    rng = npr if seed is None else npr.RandomState(seed=seed)
    zdim = get_zdim(decoder_params)

    if vecs is not None:
        v0, v1 = vecs
    elif axes is not None:
        v0, v1 = np.eye(zdim)[axes[0]], np.eye(zdim)[axes[1]]
    else:
        v0, v1 = np.linalg.qr(rng.randn(zdim, 2))[0].T

    x0, x1, y0, y1 = limits[0]*v0, limits[1]*v0, limits[2]*v1, limits[3]*v1
    interval = np.linspace(0, 1, sidelen, endpoint=True)
    regular_grid = lambda zdim: np.vstack(
            [(1-t)*x0 + t*x1 + (1-s)*y0 + s*y1 for t in interval for s in interval])
    imagevecs = points_to_imagevecs(regular_grid, decoder_params, decoder)
    return make_grid(sidelen, imagevecs, imshape)


def points_to_imagevecs(points, decoder_params, decoder=gaussian_decoder):
    decode = decoder(decoder_params)
    vals = decode(points)
    out = vals[0] if isinstance(vals, tuple) else vals
    if not isinstance(vals, np.ndarray):
        out = out.eval()
    return out


def random_grid(sidelen, decoder_params, imshape, seed=None):
    rng = npr if seed is None else npr.RandomState(seed=seed)
    zdim = get_zdim(decoder_params)

    points = rng.randn(sidelen**2, zdim)
    imagevecs = points_to_imagevecs(points, decoder_params)
    return make_grid(sidelen, imagevecs, imshape)


class Interactive(object):
    def __init__(self, draw_func, init_image, limits):
        self.fig, (self.ax, self.imax) = fig, (ax, imax) = \
            plt.subplots(1, 2, figsize=(5, 2.5), facecolor='white')
        self.canvas = canvas = fig.canvas

        self.draw_func = draw_func

        self._init_controlaxis(ax, limits)
        self._init_imageaxis(imax, init_image)
        fig.tight_layout()

        canvas.mpl_connect('draw_event', self.draw)
        canvas.mpl_connect('button_press_event', self.button_press)
        canvas.mpl_connect('button_release_event', self.button_release)
        canvas.mpl_connect('motion_notify_event', self.motion_notify)
        # TODO add resize event

    ### initialization

    def _init_controlaxis(self, ax, limits):
        ax.axis(limits)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.autoscale(False)
        self.circle = circle = Circle(
            (0,0), radius=0.05, facecolor='r', animated=True)
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


def run_interactive(decoder_params, dims, limits):
    zdim = get_zdim(decoder_params)
    decode = numpy_gaussian_decoder(decoder_params)
    vec = np.zeros(zdim)

    def draw(x, y):
        vec[dims] = (x,y)
        return reshape_square(decode(vec))

    return Interactive(draw, draw(0,0), limits)
