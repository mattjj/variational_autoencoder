from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
import sys

from scipy.misc import imresize
from scipy.ndimage import zoom

def printnow(s):
    print s
    sys.stdout.flush()

# with open('data/sod1-unsmoothed.pkl', 'r') as infile:
#     f = pickle.load(infile)

def shrink(d):
    d = d[:,15:-15,15:-15]
    d = d[~np.isnan(d.reshape(d.shape[0], -1)).any(1)]
    d = zoom(d, (1., 30./d.shape[1], 30./d.shape[2]))
    return d

shrunk = shrink(np.load('data/sod1-new.npy'))
np.save('data/sod1-new-shrunk.npy', shrunk)



# with open('data/sod1-unsmoothed.pkl', 'r') as infile:
#     f = pickle.load(infile)

# def load_image_data(key):
#     d = f[key]
#     d = d[:,15:-15,15:-15]
#     d = d[~np.isnan(d.reshape(d.shape[0], -1)).any(1)]
#     d = zoom(d, (1., 30./d.shape[1], 30./d.shape[2]))
#     printnow('done with {}'.format(key))
#     return d

# all_images = np.concatenate(map(load_image_data, f.keys()))
# np.save('data/sod1-shrunk.npy', all_images)
