#!/usr/bin/env ipython
from __future__ import division
import cPickle as pickle
import gzip
import sys
import matplotlib.pyplot as plt

from vae.viz import run_interactive

if __name__ == "__main__":
    with gzip.open('vae_fit.pkl.gz') as infile:
        encoder_params, decoder_params = pickle.load(infile)

    I = run_interactive(decoder_params, seed=27, limits=[-3, 3, -3, 3])  # seed = 1, 23, 27
    I.fig.show()
    I.fig.canvas.start_event_loop_default()
