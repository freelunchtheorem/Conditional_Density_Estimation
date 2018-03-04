#
# code skeleton from https://github.com/janvdvegt/KernelMixtureNetwork
# this version additionally supports fit_by_crossval and multidimentional Y
#

import math
import numpy as np
import tensorflow as tf
import edward as ed
from edward.models import Categorical, Mixture, Normal, MultivariateNormalDiag
from keras.layers import Dense, Dropout
import matplotlib.pyplot as plt

from .helpers import sample_center_points
from .base import BaseDensityEstimator

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)





class KernelMixtureNetwork(BaseDensityEstimator):

    def __init__(self, center_sampling_method='k_means', n_centers=20, keep_edges=False,
                 init_scales='default', estimator=None, X_ph=None, train_scales=False, n_training_epochs=300):
        """
        Main class for Kernel Mixture Network
        Args:
            center_sampling_method: String that describes the method to use for finding kernel centers
            n_centers: Number of kernels to use in the output
            keep_edges: Keep the extreme y values as center to keep expressiveness
            init_scales: List or scalar that describes (initial) values of bandwidth parameter
            estimator: Keras or tensorflow network that ends with a dense layer to place kernel mixture output on top off,
                       if None use a standard 15 -> 15 Dense network
            X_ph: Placeholder for input to your custom estimator, currently only supporting one input placeholder,
                  but should be easy to extend to a list of placeholders
            train_scales: Boolean that describes whether or not to make the scales trainable
        """

        self.sess = ed.get_session()
        self.inference = None

        self.estimator = estimator
        self.X_ph = X_ph

        self.n_training_epochs = n_training_epochs

        self.center_sampling_method = center_sampling_method
        self.n_centers = n_centers
        self.keep_edges = keep_edges

        self.train_loss = np.empty(0)
        self.test_loss = np.empty(0)

        if init_scales == 'default':
            init_scales = np.array([1])

        self.n_scales = len(init_scales)
        # Transform scales so that the softplus will result in passed init_scales
        self.init_scales = [math.log(math.exp(s) - 1) for s in init_scales]
        self.train_scales = train_scales

        self.fitted = False
        self.can_sample = True

    def fit(self, X, Y, **kwargs):
        """
        builds the Kernel Density Network model and fits the parameters by minimizing the negative
        log-likelihood of the provided data
        :param X: nummpy array to be conditioned on - shape: (n_samples, n_dim_x)
        :param Y: nummpy array of y targets - shape: (n_samples, n_dim_y)
        :param n_epoch: positive integer denoting the number of training epochs that shall be performed for fitting the model
        """

        X, Y = self._handle_input_dimensionality(X, Y, fitting=True)

        # define the full model
        self._build_model(X, Y)

        # setup inference procedure
        self.inference = ed.MAP(data={self.mixtures: self.y_ph})
        self.inference.initialize(var_list=tf.trainable_variables(), n_iter=self.n_training_epochs)
        tf.global_variables_initializer().run()

        # train the model
        self._partial_fit(X, Y, n_epoch=self.n_training_epochs, **kwargs)
        self.fitted = True

    def _partial_fit(self, X, Y, n_epoch=1, eval_set=None):
        """
        update model
        """
        print("fitting model")

        # loop over epochs
        for i in range(n_epoch):

            # run inference, update trainable variables of the model
            info_dict = self.inference.update(feed_dict={self.X_ph: X, self.y_ph: Y})

            train_loss = info_dict['loss'] / len(Y)
            self.train_loss = np.append(self.train_loss, -train_loss)

            if eval_set is not None:
                X_test, y_test = eval_set
                test_loss = self.sess.run(self.inference.loss, feed_dict={self.X_ph: X_test, self.y_ph: y_test}) / len(y_test)
                self.test_loss = np.append(self.test_loss, -test_loss)

            # only print progress for the initial fit, not for additional updates
            if not self.fitted:
                self.inference.print_progress(info_dict)

        print("mean log-loss train: {:.3f}".format(train_loss))
        if eval_set is not None:
            print("man log-loss test: {:.3f}".format(test_loss))

        print("optimal scales: {}".format(self.sess.run(self.scales)))

    def predict(self, X, Y):
        """
        computes the conditional likelihood p(y|x) given the fitted model
        :param X: nummpy array to be conditioned on - shape: (n_query_samples, n_dim_x)
        :param Y: nummpy array of y targets - shape: (n_query_samples, n_dim_y)
        :return: numpy array of shape (n_query_samples, ) holding the conditional likelihood p(y|x)
        """
        assert self.fitted, "model must be fitted to compute likelihood score"

        X, Y = self._handle_input_dimensionality(X, Y, fitting=False)
        return self.sess.run(self.likelihoods, feed_dict={self.X_ph: X, self.y_ph: Y})

    def predict_density(self, X, Y=None, resolution=100):
        """
        conditional density p(y|x) over a predefined grid of target values
        :param X values/vectors to be conditioned on - shape: (n_instances, n_dim_x)
        :param (optional) Y - y values to be evaluated from p(y|x) -  if not set, Y will be a grid with with specified resolution
        :param resulution of evaluation grid
        :return density p(y|x) shape: (n_instances, resolution**n_dim_y), Y - grid with with specified resolution - shape: (resolution**n_dim_y, n_dim_y)
        """
        if Y is None:
            max_scale = np.max(self.sess.run(self.scales))
            Y = np.linspace(self.y_min - 2.5 * max_scale, self.y_max + 2.5 * max_scale, num=resolution)
        X = self._handle_input_dimensionality(X)
        return self.sess.run(self.densities, feed_dict={self.X_ph: X, self.y_grid_ph: Y})

    def sample(self, X):
        """
        sample from the conditional mixture distributions
         :param X values/vectors to be conditioned on - shape: (n_instances, n_dim_x)
        """
        assert self.fitted, "model must be fitted to compute likelihood score"
        X = self._handle_input_dimensionality(X)
        return X, self.sess.run(self.samples, feed_dict={self.X_ph: X})

    def _build_model(self, X, Y):
        """
        implementation of the KMN
        """
        # create a placeholder for the target
        self.y_ph = y_ph = tf.placeholder(tf.float32, [None, self.ndim_y])
        self.n_sample_ph = tf.placeholder(tf.int32, None)

        # if no external estimator is provided, create a default neural network
        if self.estimator is None:
            self.X_ph = tf.placeholder(tf.float32, [None, self.ndim_x])
            # two dense hidden layers with 15 nodes each
            x = Dense(15, activation='elu')(self.X_ph)
            x = Dense(15, activation='elu')(x)
            self.estimator = x

        # get batch size
        self.batch_size = tf.shape(self.X_ph)[0]

        # locations of the gaussian kernel centers
        n_locs = self.n_centers
        self.locs = locs = sample_center_points(Y, method=self.center_sampling_method, k=n_locs, keep_edges=self.keep_edges)
        self.locs_array = locs_array = tf.unstack(tf.transpose(tf.multiply(tf.ones((self.batch_size, n_locs, self.ndim_y)), locs), perm=[1,0,2]))

        # scales of the gaussian kernels
        self.scales = scales = tf.nn.softplus(tf.Variable(self.init_scales, dtype=tf.float32, trainable=self.train_scales))
        self.scales_array = scales_array = tf.unstack(tf.transpose(tf.multiply(tf.ones((self.batch_size, self.ndim_y, self.n_scales)), scales), perm=[2,0,1]))

        # kernel weights, as output by the neural network
        self.weights = weights = Dense(n_locs * self.n_scales, activation='softplus')(self.estimator)

        # mixture distributions
        self.cat = cat = Categorical(logits=weights)
        self.components = components = [MultivariateNormalDiag(loc=loc, scale_diag=scale) for loc in locs_array for scale in scales_array]
        self.mixtures = mixtures = Mixture(cat=cat, components=components, value=tf.zeros_like(y_ph))

        # tensor to store samples
        self.samples = mixtures.sample()

        self.y_min = Y.min()
        self.y_max = Y.max()

        # placeholder for the grid
        self.y_grid_ph = y_grid_ph = tf.placeholder(tf.float32)
        # tensor to store grid point densities
        self.densities = tf.transpose(mixtures.prob(tf.reshape(y_grid_ph, (-1, 1))))

        # tensor to compute likelihoods
        self.likelihoods = mixtures.prob(y_ph)

    def plot_loss(self):
        """
        plot train loss and optionally test loss over epochs
        source: http://edwardlib.org/tutorials/mixture-density-network
        """
        # new figure
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 3))

        # plot train loss
        plt.plot(np.arange(len(self.train_loss)), self.train_loss, label='Train')

        if len(self.test_loss) > 0:
            # plot test loss
            plt.plot(np.arange(len(self.test_loss)), self.test_loss, label='Test')

        plt.legend(fontsize=20)
        plt.xlabel('epoch', fontsize=15)
        plt.ylabel('mean negative log-likelihood', fontsize=15)
        plt.show()

        return fig, axes

    def _param_grid(self):
        n_centers = [int(self.n_samples / 10), 50, 20, 10, 5]

        param_grid = {
            "n_centers": n_centers,
            "center_sampling_method": ["agglomerative", "k_means", "random"],
            "keep_edges": [True, False]
        }
        return param_grid

    def fit_by_cv(self, X, Y, n_folds=5):
        raise NotImplementedError

    def __str__(self):
        return "\nEstimator type: {}\n center sampling method: {}\n n_centers: {}\n keep_edges: {}\n init_scales: {}\n train_scales: {}\n " \
               "n_training_epochs: {}\n".format(self.__class__.__name__, self.center_sampling_method, self.n_centers, self.keep_edges,
                                                    self.init_scales, self.train_scales, self.n_training_epochs)

    def __unicode__(self):
        return self.__str__()

    def __reduce__(self):
       return (self.__class__, (self.center_sampling_method, self.n_centers, self.keep_edges,
                'default', None, None, False, 300))
