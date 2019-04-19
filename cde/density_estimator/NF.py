import numpy as np
import tensorflow as tf

import cde.utils.tf_utils.layers as L
from cde.utils.tf_utils.layers_powered import LayersPowered
from cde.utils.tf_utils.network import MLP
from .BaseDensityEstimator import BaseDensityEstimator
from .normalizing_flows import FLOWS
from cde.utils.serializable import Serializable


class NormalizingFlowEstimator(LayersPowered, Serializable, BaseDensityEstimator):
    """ Normalizing Flow Estimator

    Building on "Normalizing Flows", Rezende & Mohamed 2015

    Args:
        name: (str) name space of the network (should be unique in code, otherwise tensorflow namespace collitions may arise)
        ndim_x: (int) dimensionality of x variable
        ndim_y: (int) dimensionality of y variable
        flows_type: (tuple of string) The types of flows to use. Applied in order, going from the normal base distribution to the transformed distribution. Individual strings can be 'planar', 'radial'.
        hidden_sizes: (tuple of int) sizes of the hidden layers of the neural network
        hidden_nonlinearity: (tf function) nonlinearity of the hidden layers
        n_training_epochs: (int) Number of epochs for training
        x_noise_std: (optional) standard deviation of Gaussian noise over the the training data X -> regularization through noise
        y_noise_std: (optional) standard deviation of Gaussian noise over the the training data Y -> regularization through noise
        weight_normalization: (boolean) whether weight normalization shall be used
        data_normalization: (boolean) whether to normalize the data (X and Y) to exhibit zero-mean and std
        random_seed: (optional) seed (int) of the random number generators used
    """

    def __init__(self, name, ndim_x, ndim_y, flows_type=('affine', 'radial', 'radial', 'radial'),
                 hidden_sizes=(16, 16), hidden_nonlinearity=tf.tanh, n_training_epochs=1000, x_noise_std=None,
                 y_noise_std=None, weight_normalization=True, data_normalization=True, random_seed=None):
        Serializable.quick_init(self, locals())
        self._check_uniqueness_of_scope(name)
        assert all([f in FLOWS.keys() for f in flows_type])

        self.name = name
        self.ndim_x = ndim_x
        self.ndim_y = ndim_y

        self.random_seed = random_seed
        self.random_state = np.random.RandomState(seed=random_seed)
        tf.set_random_seed(random_seed)

        # charateristics of the flows to be used
        self.n_flows = len(flows_type)
        self.flows_type = flows_type

        # specification of the network
        self.hidden_sizes = hidden_sizes
        self.hidden_nonlinearity = hidden_nonlinearity

        self.n_training_epochs = n_training_epochs

        # regularization parameters, currently not used
        self.x_noise_std = x_noise_std
        self.y_noise_std = y_noise_std

        # normalizing the network weights
        self.weight_normalization = weight_normalization

        # whether to normalize the data to zero mean, and uniform variance
        self.data_normalization = data_normalization

        # as we'll be using reversed flows, sampling is too slow to be useful
        self.can_sample = False
        self.has_pdf = True
        self.has_cdf = True

        self.fitted = False

        # build tensorflow model
        self._build_model()

    def fit(self, X, Y, random_seed=None, verbose=True, eval_set=None, **kwargs):
        """
        Fit the model with to the provided data

        :param X: numpy array to be conditioned on - shape: (n_samples, n_dim_x)
        :param Y: numpy array of y targets - shape: (n_sample, n_dim_y)
        :param eval_set: (tuple) eval/test set - tuple (X_test, Y_test)
        :param verbose: (boolean) controls the verbosity of console output
        """

        X, Y = self._handle_input_dimensionality(X, Y, fitting=True)

        if tf.get_default_session():
            self.sess = tf.get_default_session()
        else:
            self.sess = tf.Session()
            self.sess.as_default()

        if eval_set:
            eval_set = tuple(self._handle_input_dimensionality(x) for x in eval_set)

        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        tf.initializers.variables(var_list, name='init').run(session=self.sess)

        if self.data_normalization:
            self._compute_data_normalization(X, Y)

        for i in range(0, self.n_training_epochs + 1):
            self.sess.run(self.train_step, feed_dict={self.X_ph: X, self.Y_ph: Y, self.train_phase: True})
            if verbose and not i % 100:
                log_loss = self.sess.run(self.log_loss, feed_dict={self.X_ph: X, self.Y_ph: Y})
                if not eval_set:
                    print('Step {:4}: train log-loss {: .4f}'.format(i, log_loss))
                else:
                    eval_ll = self.sess.run(self.log_loss, feed_dict={self.X_ph: eval_set[0], self.Y_ph: eval_set[1]})
                    print('Step {:4}: train log-loss {: .4f} eval log-loss {: .4f}'.format(i, log_loss, eval_ll))

        self.fitted = True

    def pdf(self, X, Y):
        """ Predicts the conditional probability p(y|x). Requires the model to be fitted

        :param X: numpy array to be conditioned on - shape: (n_samples, n_dim_x)
        :param Y: numpy array of y targets - shape: (n_samples, n_dim_y)
        :return: conditional probability p(y|x) - numpy array of shape (n_query_samples, )
        """
        assert self.fitted, "model must be fitted to evaluate the likelihood score"

        X, Y = self._handle_input_dimensionality(X, Y, fitting=False)
        p = self.sess.run(self.pdf_, feed_dict={self.X_ph: X, self.Y_ph: Y})
        assert p.ndim == 1, "N_dim should be 1, is {}".format(p.ndim)
        assert p.shape[0] == X.shape[0], "Shapes should be equal, are {} != {}".format(p.shape[0], X.shape[0])
        return p

    def log_pdf(self, X, Y):
        """ Predicts the log of the conditional probability p(y|x). Requires the model to be fitted

        :param X: numpy array to be conditioned on - shape: (n_samples, n_dim_x)
        :param Y: numpy array of y targets - shape: (n_samples, n_dim_y)
        :return: log of the conditional probability p(y|x) - numpy array of shape (n_query_samples, )
        """
        assert self.fitted, "model must be fitted to evaluate the likelihood score"

        X, Y = self._handle_input_dimensionality(X, Y, fitting=False)
        p = self.sess.run(self.log_pdf_, feed_dict={self.X_ph: X, self.Y_ph: Y})
        assert p.ndim == 1, "N_dim should be 1, is {}".format(p.ndim)
        assert p.shape[0] == X.shape[0], "Shapes should be equal, are {} != {}".format(p.shape[0], X.shape[0])
        return p

    def cdf(self, X, Y):
        """ Predicts the conditional cumulative probability p(Y<=y|X=x). Requires the model to be fitted.

        :param X: numpy array to be conditioned on - shape: (n_samples, n_dim_x)
        :param Y: numpy array of y targets - shape: (n_samples, n_dim_y)
        :return: conditional cumulative probability p(Y<=y|X=x) - numpy array of shape (n_query_samples, )
        """
        assert self.fitted, "model must be fitted to evaluate the likelihood score"
        X, Y = self._handle_input_dimensionality(X, Y, fitting=False)
        p = self.sess.run(self.cdf_, feed_dict={self.X_ph: X, self.Y_ph: Y})
        assert p.ndim == 1, "N_dim should be 1, is {}".format(p.ndim)
        assert p.shape[0] == X.shape[0], "Shapes should be equal, are {} != {}".format(p.shape[0], X.shape[0])
        return p

    def _build_model(self):
        """
        implementation of the flow model
        """
        with tf.variable_scope(self.name):
            # adds placeholders, data normalization and data noise to graph as desired
            self.layer_in_x, self.layer_in_y = self._build_input_layers()
            self.y_input = L.get_output(self.layer_in_y)

            flow_classes = [FLOWS[flow_name] for flow_name in self.flows_type]
            # get the individual parameter sizes for each flow
            param_split_sizes = [flow.get_param_size(self.ndim_y) for flow in flow_classes]
            mlp_output_dim = sum(param_split_sizes)
            core_network = MLP(
                name="core_network",
                input_layer=self.layer_in_x,
                output_dim=mlp_output_dim,
                hidden_sizes=self.hidden_sizes,
                hidden_nonlinearity=self.hidden_nonlinearity,
                output_nonlinearity=None,
                weight_normalization=self.weight_normalization
            )
            outputs = L.get_output(core_network.output_layer)
            flow_params = tf.split(value=outputs, num_or_size_splits=param_split_sizes, axis=1)

            # instanciate the flows with their parameters
            flows = [flow(params, self.ndim_y) for flow, params in zip(flow_classes, flow_params)]

            # build up the base distribution that will be transformed by the flows
            base_dist = tf.distributions.Normal(loc=[0.]*self.ndim_y, scale=[1.]*self.ndim_y)

            # chain the flows together and build the transformed distribution using the base_dist + flows
            # Chaining applies the flows in reverse, Chain([a,b]).forward(x) being a.forward(b.forward(x))
            # We reverse them so the flows are stacked ontop of the base distribution in the original order
            flows.reverse()
            chain = tf.contrib.distributions.bijectors.Chain(flows)
            target_dist = tf.contrib.distributions.TransformedDistribution(distribution=base_dist, bijector=chain)

            # since we operate with matrices not vectors, the output would have dimension (?,1)
            # and therefor has to be reduce first to have shape (?,)
            if self.data_normalization:
                self.pdf_ = tf.squeeze(target_dist.prob(self.y_input) / tf.reduce_prod(self.std_y_sym), axis=1)
                self.log_pdf_ = tf.squeeze(target_dist.log_prob(self.y_input) - tf.reduce_sum(tf.log(self.std_y_sym)), axis=1)
                self.cdf_ = tf.squeeze(target_dist.cdf(self.y_input) / tf.reduce_prod(self.std_y_sym), axis=1)
            else:
                self.pdf_ = tf.squeeze(target_dist.prob(self.y_input), axis=1)
                self.log_pdf_ = tf.squeeze(target_dist.log_prob(self.y_input), axis=1)
                self.cdf_ = tf.squeeze(target_dist.cdf(self.y_input), axis=1)

            self.loss = -tf.reduce_prod(self.pdf_)
            self.log_loss = -tf.reduce_sum(self.log_pdf_)
            self.train_step = tf.train.AdamOptimizer().minimize(self.log_loss)

        # initialize LayersPowered -> provides functions for serializing tf models
        LayersPowered.__init__(self, [self.layer_in_y, core_network.output_layer])


    def _build_input_layers(self):
        # Input_Layers & placeholders
        self.X_ph = tf.placeholder(tf.float32, shape=(None, self.ndim_x))
        self.Y_ph = tf.placeholder(tf.float32, shape=(None, self.ndim_y))
        self.train_phase = tf.placeholder_with_default(False, None)

        layer_in_x = L.InputLayer(shape=(None, self.ndim_x), input_var=self.X_ph, name="input_x")
        layer_in_y = L.InputLayer(shape=(None, self.ndim_y), input_var=self.Y_ph, name="input_y")

        # add data normalization layer if desired
        if self.data_normalization:
            layer_in_x = L.NormalizationLayer(layer_in_x, self.ndim_x, name="data_norm_x")
            self.mean_x_sym, self.std_x_sym = layer_in_x.get_params()
            layer_in_y = L.NormalizationLayer(layer_in_y, self.ndim_y, name="data_norm_y")
            self.mean_y_sym, self.std_y_sym = layer_in_y.get_params()

        # add noise layer if desired
        if self.x_noise_std is not None:
            layer_in_x = L.GaussianNoiseLayer(layer_in_x, self.x_noise_std, noise_on_ph=self.train_phase)
        if self.y_noise_std is not None:
            layer_in_y = L.GaussianNoiseLayer(layer_in_y, self.y_noise_std, noise_on_ph=self.train_phase)

        return layer_in_x, layer_in_y

    def _compute_data_normalization(self, X, Y):
        # compute data statistics (mean & std)
        self.x_mean = np.mean(X, axis=0)
        self.x_std = np.std(X, axis=0)
        self.y_mean = np.mean(Y, axis=0)
        self.y_std = np.std(Y, axis=0)

        self.data_statistics = {
            'X_mean': self.x_mean,
            'X_std': self.x_std,
            'Y_mean': self.y_mean,
            'Y_std': self.y_std,
        }

        # assign them to tf variables
        self.sess.run([
            tf.assign(self.mean_x_sym, self.x_mean),
            tf.assign(self.std_x_sym, self.x_std),
            tf.assign(self.mean_y_sym, self.y_mean),
            tf.assign(self.std_y_sym, self.y_std)
        ])

    @staticmethod
    def _check_uniqueness_of_scope(name):
        current_scope = tf.get_variable_scope().name
        scopes = set([variable.name.split('/')[0] for variable in tf.global_variables(scope=current_scope)])
        assert name not in scopes, "%s is already in use for a tensorflow scope - please choose another estimator name"%name

    def __getstate__(self):
        state = LayersPowered.__getstate__(self)
        state['fitted'] = self.fitted
        return state

    def __setstate__(self, state):
        LayersPowered.__setstate__(self, state)
        self.fitted = state['fitted']
        self.sess = tf.get_default_session()

    def __str__(self):
        return "\nEstimator type: {}\n n_flows: {}\n flows_type: {}\n entropy_reg_coef: {}\n data_normalization: {} \n weight_normalization: {}\n" \
               "n_training_epochs: {}\n x_noise_std: {}\n y_noise_std: {}\n ".format(self.__class__.__name__, self.n_flows, self.flows_type, self.entropy_reg_coef,
                                                                                 self.data_normalization, self.weight_normalization, self.n_training_epochs, self.x_noise_std, self.y_noise_std)
