import numpy as np
import tensorflow as tf
from edward.models import Categorical, Mixture, MultivariateNormalDiag
from cde.utils.tf_utils.network import MLP
import cde.utils.tf_utils.layers as L
from cde.utils.tf_utils.layers_powered import LayersPowered
from cde.utils.serializable import Serializable


from .BaseNNMixtureEstimator import BaseNNMixtureEstimator

class MixtureDensityNetwork(BaseNNMixtureEstimator):
  """ Mixture Density Network Estimator

    See "Mixture Density networks", Bishop 1994

    Args:
        name: (str) name space of MDN (should be unique in code, otherwise tensorflow namespace collitions may arise)
        ndim_x: (int) dimensionality of x variable
        ndim_y: (int) dimensionality of y variable
        n_centers: Number of Gaussian mixture components
        hidden_sizes: (tuple of int) sizes of the hidden layers of the neural network
        hidden_nonlinearity: (tf function) nonlinearity of the hidden layers
        n_training_epochs: Number of epochs for training
        x_noise_std: (optional) standard deviation of Gaussian noise over the the training data X -> regularization through noise
        y_noise_std: (optional) standard deviation of Gaussian noise over the the training data Y -> regularization through noise
        adaptive_noise_fn: (callable) that takes the number of samples and the data dimensionality as arguments and returns
                                   the noise std as float - if used, the x_noise_std and y_noise_std have no effect
        entropy_reg_coef: (optional) scalar float coefficient for shannon entropy penalty on the mixture component weight distribution
        weight_decay: (float) the amount of decoupled (http://arxiv.org/abs/1711.05101) weight decay to apply
        l2_reg: (float) the amount of l2 penalty on neural network weights
        l1_reg: (float) the amount of l1 penalty on neural network weights
        weight_normalization: (boolean) whether weight normalization shall be used
        data_normalization: (boolean) whether to normalize the data (X and Y) to exhibit zero-mean and std
        dropout: (float) the probability of switching off nodes during training
        random_seed: (optional) seed (int) of the random number generators used
    """


  def __init__(self, name, ndim_x, ndim_y, n_centers=10, hidden_sizes=(16, 16), hidden_nonlinearity=tf.nn.tanh,
               n_training_epochs=1000, x_noise_std=None, y_noise_std=None, adaptive_noise_fn=None, entropy_reg_coef=0.0,
               weight_decay=0.0, weight_normalization=True, data_normalization=True, dropout=0.0, l2_reg=0.0, l1_reg=0.0,
               random_seed=None):

    Serializable.quick_init(self, locals())
    self._check_uniqueness_of_scope(name)

    self.name = name
    self.ndim_x = ndim_x
    self.ndim_y = ndim_y

    self.random_seed = random_seed
    self.random_state = np.random.RandomState(seed=random_seed)
    tf.set_random_seed(random_seed)

    self.n_centers = n_centers

    self.hidden_sizes = hidden_sizes
    self.hidden_nonlinearity = hidden_nonlinearity

    self.n_training_epochs = n_training_epochs

    # regularization parameters
    self.x_noise_std = x_noise_std
    self.y_noise_std = y_noise_std
    self.entropy_reg_coef = entropy_reg_coef
    self.adaptive_noise_fn = adaptive_noise_fn
    self.weight_decay = weight_decay
    self.l2_reg = l2_reg
    self.l1_reg = l1_reg
    self.weight_normalization = weight_normalization
    self.data_normalization = data_normalization
    self.dropout = dropout

    self.can_sample = True
    self.has_pdf = True
    self.has_cdf = True

    self.fitted = False

    # build tensorflow model
    self._build_model()

  def fit(self, X, Y, random_seed=None, verbose=True, eval_set=None, **kwargs):
    """ Fits the conditional density model with provided data

      Args:
        X: numpy array to be conditioned on - shape: (n_samples, n_dim_x)
        Y: numpy array of y targets - shape: (n_samples, n_dim_y)
        eval_set: (tuple) eval/test set - tuple (X_test, Y_test)
        verbose: (boolean) controls the verbosity (console output)

    """
    X, Y = self._handle_input_dimensionality(X, Y, fitting=True)

    if eval_set is not None:
      eval_set = self._handle_input_dimensionality(*eval_set)

    self._setup_inference_and_initialize()

    # data normalization if desired
    if self.data_normalization: # this must happen after the initialization
      self._compute_data_normalization(X, Y)  # computes mean & std of data and assigns it to tf graph for normalization

    self._compute_noise_intensity(X, Y)

    # train the model
    self._partial_fit(X, Y, n_epoch=self.n_training_epochs, verbose=verbose, eval_set=eval_set)
    self.fitted = True

  def _build_model(self):
    """
    implementation of the MDN
    """

    with tf.variable_scope(self.name):
      # adds placeholders, data_normalization and data_noise if desired. Also adds a placeholder for dropout probability
      self.layer_in_x, self.layer_in_y = self._build_input_layers()

      # create core multi-layer perceptron
      mlp_output_dim = 2 * self.ndim_y * self.n_centers + self.n_centers
      core_network = MLP(
              name="core_network",
              input_layer=self.layer_in_x,
              output_dim=mlp_output_dim,
              hidden_sizes=self.hidden_sizes,
              hidden_nonlinearity=self.hidden_nonlinearity,
              output_nonlinearity=None,
              weight_normalization=self.weight_normalization,
              dropout_ph=self.dropout_ph if self.dropout else None
          )

      core_output_layer = core_network.output_layer

      # slice output of MLP into three equally sized parts for loc, scale and mixture weights
      slice_layer_locs = L.SliceLayer(core_output_layer, indices=slice(0, self.ndim_y * self.n_centers), axis=-1)
      slice_layer_scales = L.SliceLayer(core_output_layer, indices=slice(self.ndim_y * self.n_centers, 2 * self.ndim_y * self.n_centers), axis=-1)
      slice_layer_weights = L.SliceLayer(core_output_layer, indices=slice(2 * self.ndim_y * self.n_centers, mlp_output_dim), axis=-1)

      # locations mixture components
      self.reshape_layer_locs = L.ReshapeLayer(slice_layer_locs, (-1, self.n_centers, self.ndim_y))
      self.locs = L.get_output(self.reshape_layer_locs)

      # scales of the mixture components
      reshape_layer_scales = L.ReshapeLayer(slice_layer_scales, (-1, self.n_centers, self.ndim_y))
      self.softplus_layer_scales = L.NonlinearityLayer(reshape_layer_scales, nonlinearity=tf.nn.softplus)
      self.scales = L.get_output(self.softplus_layer_scales)

      # weights of the mixture components
      self.logits = L.get_output(slice_layer_weights)
      self.softmax_layer_weights = L.NonlinearityLayer(slice_layer_weights, nonlinearity=tf.nn.softmax)
      self.weights = L.get_output(self.softmax_layer_weights)

      # # put mixture components together
      self.y_input = L.get_output(self.layer_in_y)
      self.cat = cat = Categorical(logits=self.logits)
      self.components = components = [MultivariateNormalDiag(loc=loc, scale_diag=scale) for loc, scale
                     in zip(tf.unstack(self.locs, axis=1), tf.unstack( self.scales, axis=1))]
      self.mixture = mixture = Mixture(cat=cat, components=components, value=tf.zeros_like(self.y_input))

      # regularization
      self._add_softmax_entropy_regularization()
      self._add_l1_l2_regularization(core_network)

      # tensor to store samples
      self.samples = mixture.sample() #TODO either use it or remove it

      # tensor to compute probabilities
      if self.data_normalization:
        self.pdf_ = mixture.prob(self.y_input) / tf.reduce_prod(self.std_y_sym)
        self.log_pdf_ = mixture.log_prob(self.y_input) - tf.reduce_sum(tf.log(self.std_y_sym))
      else:
        self.pdf_ = mixture.prob(self.y_input)
        self.log_pdf_ = mixture.log_prob(self.y_input)

      # symbolic tensors for getting the unnormalized mixture components
      if self.data_normalization:
        self.scales_unnormalized = self.scales * self.std_y_sym
        self.locs_unnormalized = self.locs * self.std_y_sym + self.mean_y_sym
      else:
        self.scales_unnormalized = self.scales
        self.locs_unnormalized = self.locs

    # initialize LayersPowered --> provides functions for serializing tf models
    LayersPowered.__init__(self, [self.softmax_layer_weights, self.softplus_layer_scales, self.reshape_layer_locs,
                                  self.layer_in_y])

  def _param_grid(self):
    param_grid = {
        "n_training_epochs": [500, 1000],
        "n_centers": [5, 10, 20],
        "x_noise_std": [0.1, 0.15, 0.2, 0.3],
        "y_noise_std": [0.1, 0.15, 0.2]
    }
    return param_grid

  def _get_mixture_components(self, X):
    assert self.fitted
    weights, locs, scales = self.sess.run([self.weights, self.locs_unnormalized, self.scales_unnormalized], feed_dict={self.X_ph: X})
    assert weights.shape[0] == locs.shape[0] == scales.shape[0] == X.shape[0]
    assert weights.shape[1] == locs.shape[1] == scales.shape[1] == self.n_centers
    assert locs.shape[2] == scales.shape[2] == self.ndim_y
    assert locs.ndim == 3 and scales.ndim == 3 and weights.ndim == 2
    return weights, locs, scales

  def __str__(self):
    return "\nEstimator type: {}\n n_centers: {}\n entropy_reg_coef: {}\n data_normalization: {} \n weight_normalization: {}\n" \
             "n_training_epochs: {}\n x_noise_std: {}\n y_noise_std: {}\n ".format(self.__class__.__name__, self.n_centers, self.entropy_reg_coef,
                                                  self.data_normalization, self.weight_normalization, self.n_training_epochs, self.x_noise_std, self.y_noise_std)