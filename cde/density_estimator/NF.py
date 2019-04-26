import numpy as np
import tensorflow as tf

from .BaseDensityEstimator import BaseDensityEstimator
from .normalizing_flows import FLOWS


class NormalizingFlowEstimator(BaseDensityEstimator):
    """ Normalizing Flow Estimator
    @todo: data normalization
    @todo: eval set
    @todo: Affine bijector

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
        entropy_reg_coef: (optional) scalar float coefficient for shannon entropy penalty on the mixture component weight distribution
        weight_normalization: (boolean) whether weight normalization shall be used
        data_normalization: (boolean) whether to normalize the data (X and Y) to exhibit zero-mean and std
        random_seed: (optional) seed (int) of the random number generators used
    """

    def __init__(self, name, ndim_x, ndim_y, flows_type=('radial', 'radial', 'radial'),
                 hidden_sizes=(16, 16), hidden_nonlinearity=tf.tanh, n_training_epochs=1000, x_noise_std=None,
                 y_noise_std=None, entropy_reg_coef=0.0, weight_normalization=True,
                 data_normalization=False, random_seed=None):
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
        self.entropy_reg_coef = entropy_reg_coef
        self.weight_normalization = weight_normalization
        self.data_normalization = data_normalization

        # as we'll be using reversed flows, sampling is too slow to be useful
        self.can_sample = False
        self.has_pdf = True
        self.has_cdf = True

        self.fitted = False
        self.sess = tf.Session()

        # build tensorflow model
        self._build_model()

    def fit(self, X, Y, random_seed=None, verbose=True, eval_set=None, **kwargs):
        """
        Fit the model with to the provided data

        :param X: numpy array to be conditioned on - shape: (n_samples, n_dim_x)
        :param Y: numpy array of y targets - shape: (n_samples, n_dim_y)
        :param eval_set: (tuple) eval/test set - tuple (X_test, Y_test)
        :param verbose: (boolean) controls the verbosity of console output
        """

        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        tf.initializers.variables(var_list, name='init').run(session=self.sess)

        for i in range(0, self.n_training_epochs + 1):
            self.sess.run(self.train_step, feed_dict={self.x_input: X, self.y_input: Y})
            if verbose and not i % 100:
                log_loss = self.sess.run(self.log_loss, feed_dict={self.x_input: X, self.y_input: Y})
                print('Step {:4}: log-loss {: .4f}'.format(i, log_loss))

        self.fitted = True

    def pdf(self, X, Y):
        """ Predicts the conditional probability p(y|x). Requires the model to be fitted

        :param X: numpy array to be conditioned on - shape: (n_samples, n_dim_x)
        :param Y: numpy array of y targets - shape: (n_samples, n_dim_y)
        :return: conditional probability p(y|x) - numpy array of shape (n_query_samples, )
        """
        assert self.fitted, "model must be fitted to evaluate the likelihood score"

        X, Y = self._handle_input_dimensionality(X, Y, fitting=False)
        p = self.sess.run(tf.squeeze(self.pdf_, axis=-1), feed_dict={self.x_input: X, self.y_input: Y})
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
        p = self.sess.run(tf.squeeze(self.log_pdf_, axis=-1), feed_dict={self.x_input: X, self.y_input: Y})
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
        p = self.sess.run(tf.squeeze(self.cdf_, axis=-1), feed_dict={self.x_input: X, self.y_input: Y})
        assert p.ndim == 1, "N_dim should be 1, is {}".format(p.ndim)
        assert p.shape[0] == X.shape[0], "Shapes should be equal, are {} != {}".format(p.shape[0], X.shape[0])
        return p

    def _build_model(self):
        """
        implementation of the flow model
        """
        with tf.variable_scope(self.name):
            self.x_input = tf.placeholder(dtype=tf.float32, shape=(None, self.ndim_x), name='x_input')
            self.y_input = tf.placeholder(dtype=tf.float32, shape=(None, self.ndim_y), name='y_input')

            flow_classes = [FLOWS[flow_name] for flow_name in self.flows_type]
            # get the individual parameter sizes for each flow
            param_split_sizes = [flow.get_param_size(self.ndim_y) for flow in flow_classes]

            # building the hidden layers
            prev_layer = self.x_input
            for hidden_size in self.hidden_sizes:
                current = tf.layers.dense(inputs=prev_layer,
                                          units=hidden_size,
                                          activation=self.hidden_nonlinearity,
                                          kernel_regularizer=tf.keras.regularizers.l2())
                prev_layer = current
            # output layer
            mlp_output_dim = sum(param_split_sizes)
            outputs = tf.layers.dense(inputs=prev_layer,
                                      units=mlp_output_dim,
                                      activation=None,
                                      kernel_regularizer=tf.keras.regularizers.l2())

            # slice the output into the parameters for each flow
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

            self.pdf_ = target_dist.prob(self.y_input)
            self.log_pdf_ = target_dist.log_prob(self.y_input)
            self.cdf_ = target_dist.cdf(self.y_input)

            self.loss = -tf.reduce_mean(target_dist.prob(self.y_input))
            self.log_loss = -tf.reduce_mean(target_dist.log_prob(self.y_input))
            self.train_step = tf.train.AdamOptimizer().minimize(self.log_loss)

    @staticmethod
    def _check_uniqueness_of_scope(name):
        current_scope = tf.get_variable_scope().name
        scopes = set([variable.name.split('/')[0] for variable in tf.global_variables(scope=current_scope)])
        assert name not in scopes, "%s is already in use for a tensorflow scope - please choose another estimator name"%name

    def __str__(self):
        return "\nEstimator type: {}\n n_flows: {}\n flows_type: {}\n entropy_reg_coef: {}\n data_normalization: {} \n weight_normalization: {}\n" \
               "n_training_epochs: {}\n x_noise_std: {}\n y_noise_std: {}\n ".format(self.__class__.__name__, self.n_flows, self.flows_type, self.entropy_reg_coef,
                                                                                 self.data_normalization, self.weight_normalization, self.n_training_epochs, self.x_noise_std, self.y_noise_std)
