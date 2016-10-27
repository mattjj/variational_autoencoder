**NOTE** There's a [better TensorFlow version currently in the SVAE repo](https://github.com/mattjj/svae/blob/a736c3774967b003e338d0e64ae50afc1e5e30b5/examples/vae.py). The Python code is cleaner (both in that example file and in [tf_nnet.py](https://github.com/mattjj/svae/blob/a736c3774967b003e338d0e64ae50afc1e5e30b5/svae/tf_nnet.py)), so I recommend looking there instead of here.

This implementation generally follows the style in [these Theano
tutorials](https://github.com/Newmu/Theano-Tutorials) but loads all the training
data onto the GPU as in [other Theano
examples](http://deeplearning.net/tutorial/code/logistic_sgd.py). See also an
earlier [VAE implementation](https://github.com/y0ast/Variational-Autoencoder).
