from __future__ import division
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap

from nnet import compose, numpy_tanh_layer, numpy_linear_layer
from util import sigmoid, reshape_square, get_zdim, unpack_gaussian_params


#####################
#  encoder/decoder  #
#####################

get_zdim = lambda decoder_params: decoder_params[0][0].shape[0]


def gaussian_decoder(decoder_params):
    # mostly redundant code with encoder and gaussian_decoder in vae.py
    nnet_params, (W_mu, b_mu) = decoder_params[:-2], decoder_params[-2]
    nnet = compose(numpy_tanh_layer(W, b) for W, b in nnet_params)
    mu = numpy_linear_layer(W_mu, b_mu)

    def decode(X):
        return sigmoid(mu(nnet(X)))

    return decode


def encoder(encoder_params, tanh_scale):
    # mostly redundant code with encoder and gaussian_decoder in vae.py
    nnet_params, (W_h, b_h), (W_J, b_J) = \
        encoder_params[:-2], encoder_params[-2], encoder_params[-1]

    nnet = compose(numpy_tanh_layer(W, b) for W, b in nnet_params)
    h = numpy_linear_layer(W_h, b_h)
    log_J = numpy_linear_layer(W_J, b_J)

    def encode(X):
        nnet_outputs = nnet(X)
        J = -1./2 * np.exp(tanh_scale * np.tanh(log_J(nnet_outputs) / tanh_scale))
        return J, h(nnet_outputs)

    return encode


##########################
#  plotting image grids  #
##########################

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
    plt.grid(True, color='w', linestyle='-')

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


#################
#  interactive  #
#################

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


def numpy_encoder(encoder_params, tanh_scale):
    # mostly redundant code with encoder in vae.py
    nnet_params, (W_h, b_h), (W_J, b_J) = \
        unpack_gaussian_params(encoder_params)

    nnet = compose(numpy_tanh_layer(W, b) for W, b in nnet_params)
    h = numpy_linear_layer(W_h, b_h)
    log_J = numpy_linear_layer(W_J, b_J)

    def encode(X):
        nnet_outputs = nnet(X)
        J = -1./2 * np.exp(tanh_scale * np.tanh(log_J(nnet_outputs) / tanh_scale))
        return J, h(nnet_outputs)

    return encode

def run_interactive(decoder_params, dims=None, seed=None, limits=[-3., 3., -3., 3.]):
    zdim = get_zdim(decoder_params)
    decode = numpy_gaussian_decoder(decoder_params)

    if not (dims is None) ^ (seed is None):
        raise ValueError

    if dims is not None:
        out = np.zeros(zdim)
        def vec(x, y):
            out[dims] = x, y
            return out
    else:
        basis = np.linalg.qr(npr.RandomState(seed).randn(zdim, 2))[0]
        vec = lambda x, y: np.dot(basis, (x, y))

    draw = lambda x, y: reshape_square(decode(vec(x, y)))
    return Interactive(draw, draw(0,0), limits)


###############
#  colormaps  #
###############

def register_parula1():
    cmap = LinearSegmentedColormap.from_list(
        'parula',
        ['#352A87',
        '#0268E1',
        '#108ED2',
        # '#0FAEB9',  # maybe
        '#65BE86',  # maybe
        '#C0BC60',
        '#FFC337',
        '#F9FB0E'])
    cm.register_cmap(name='parula', cmap=cmap)

def register_parula2():
    cm_data = [[0.2081, 0.1663, 0.5292], [0.2116238095, 0.1897809524, 0.5776761905], 
    [0.212252381, 0.2137714286, 0.6269714286], [0.2081, 0.2386, 0.6770857143], 
    [0.1959047619, 0.2644571429, 0.7279], [0.1707285714, 0.2919380952, 
    0.779247619], [0.1252714286, 0.3242428571, 0.8302714286], 
    [0.0591333333, 0.3598333333, 0.8683333333], [0.0116952381, 0.3875095238, 
    0.8819571429], [0.0059571429, 0.4086142857, 0.8828428571], 
    [0.0165142857, 0.4266, 0.8786333333], [0.032852381, 0.4430428571, 
    0.8719571429], [0.0498142857, 0.4585714286, 0.8640571429], 
    [0.0629333333, 0.4736904762, 0.8554380952], [0.0722666667, 0.4886666667, 
    0.8467], [0.0779428571, 0.5039857143, 0.8383714286], 
    [0.079347619, 0.5200238095, 0.8311809524], [0.0749428571, 0.5375428571, 
    0.8262714286], [0.0640571429, 0.5569857143, 0.8239571429], 
    [0.0487714286, 0.5772238095, 0.8228285714], [0.0343428571, 0.5965809524, 
    0.819852381], [0.0265, 0.6137, 0.8135], [0.0238904762, 0.6286619048, 
    0.8037619048], [0.0230904762, 0.6417857143, 0.7912666667], 
    [0.0227714286, 0.6534857143, 0.7767571429], [0.0266619048, 0.6641952381, 
    0.7607190476], [0.0383714286, 0.6742714286, 0.743552381], 
    [0.0589714286, 0.6837571429, 0.7253857143], 
    [0.0843, 0.6928333333, 0.7061666667], [0.1132952381, 0.7015, 0.6858571429], 
    [0.1452714286, 0.7097571429, 0.6646285714], [0.1801333333, 0.7176571429, 
    0.6424333333], [0.2178285714, 0.7250428571, 0.6192619048], 
    [0.2586428571, 0.7317142857, 0.5954285714], [0.3021714286, 0.7376047619, 
    0.5711857143], [0.3481666667, 0.7424333333, 0.5472666667], 
    [0.3952571429, 0.7459, 0.5244428571], [0.4420095238, 0.7480809524, 
    0.5033142857], [0.4871238095, 0.7490619048, 0.4839761905], 
    [0.5300285714, 0.7491142857, 0.4661142857], [0.5708571429, 0.7485190476, 
    0.4493904762], [0.609852381, 0.7473142857, 0.4336857143], 
    [0.6473, 0.7456, 0.4188], [0.6834190476, 0.7434761905, 0.4044333333], 
    [0.7184095238, 0.7411333333, 0.3904761905], 
    [0.7524857143, 0.7384, 0.3768142857], [0.7858428571, 0.7355666667, 
    0.3632714286], [0.8185047619, 0.7327333333, 0.3497904762], 
    [0.8506571429, 0.7299, 0.3360285714], [0.8824333333, 0.7274333333, 0.3217], 
    [0.9139333333, 0.7257857143, 0.3062761905], [0.9449571429, 0.7261142857, 
    0.2886428571], [0.9738952381, 0.7313952381, 0.266647619], 
    [0.9937714286, 0.7454571429, 0.240347619], [0.9990428571, 0.7653142857, 
    0.2164142857], [0.9955333333, 0.7860571429, 0.196652381], 
    [0.988, 0.8066, 0.1793666667], [0.9788571429, 0.8271428571, 0.1633142857], 
    [0.9697, 0.8481380952, 0.147452381], [0.9625857143, 0.8705142857, 0.1309], 
    [0.9588714286, 0.8949, 0.1132428571], [0.9598238095, 0.9218333333, 
    0.0948380952], [0.9661, 0.9514428571, 0.0755333333], 
    [0.9763, 0.9831, 0.0538]]

    parula_map = LinearSegmentedColormap.from_list('parula', cm_data)
    cm.register_cmap(name='parula', cmap=parula_map)
