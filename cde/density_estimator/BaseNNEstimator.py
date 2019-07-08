import numpy as np
import tensorflow as tf
import sklearn
import os
import itertools
import warnings
from multiprocessing import Manager

from cde.utils.tf_utils.layers_powered import LayersPowered
import cde.utils.tf_utils.layers as L
from cde.utils.serializable import Serializable
from cde.utils.async_executor import AsyncExecutor
from cde.density_estimator.BaseDensityEstimator import BaseDensityEstimator


class BaseNNEstimator(LayersPowered, Serializable, BaseDensityEstimator):
    """
    Base class for a density estimator using a neural network to parametrize the distribution p(y|x)
    To use this class, implement pdf_, cdf_ and log_pdf_ or overwrite the parent methods
    To use the hyperparameter search, also implement reset_fit and (optionally) _param_grid()
    """

    # input data can be normalized before training
    data_normalization = False

    # used for noise regularization of the data
    x_noise_std = False
    y_noise_std = False

    # was the model fitted to the data or not
    fitted = False

    # set to >0. to use dropout during training. Determines the probability of dropping the output of a node
    dropout = 0.0

    def reset_fit(self):
        """
        Reset all tensorflow objects to enable the model to be trained again
        :return:
        """
        raise NotImplementedError()

    def fit_by_cv(self, X, Y, n_folds=3, param_grid=None, random_state=None, verbose=True, n_jobs=-1):
        """ Fits the conditional density model with hyperparameter search and cross-validation.

        - Determines the best hyperparameter configuration from a pre-defined set using cross-validation. Thereby,
          the conditional log-likelihood is used for simulation_eval.
        - Fits the model with the previously selected hyperparameter configuration

        Args:
          X: numpy array to be conditioned on - shape: (n_samples, n_dim_x)
          Y: numpy array of y targets - shape: (n_samples, n_dim_y)
          n_folds: number of cross-validation folds (positive integer)
          param_grid: (optional) a dictionary with the hyperparameters of the model as key and and a list of respective
                      parametrizations as value. The hyperparameter search is performed over the cartesian product of
                      the provided lists.
                      Example::
                              {"n_centers": [20, 50, 100, 200],
                               "center_sampling_method": ["agglomerative", "k_means", "random"],
                               "keep_edges": [True, False]
                              }
          random_state: (int) seed used by the random number generator for shuffeling the data
        """
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        original_params = self.get_params()

        if param_grid is None:
            param_grid = self._param_grid()

        param_list = list(sklearn.model_selection.ParameterGrid(param_grid))
        train_splits, test_splits = list(zip(*list(sklearn.model_selection.KFold(n_splits=n_folds, shuffle=False,
                                                                                 random_state=random_state).split(X))))

        param_ids, fold_ids = list(zip(*itertools.product(range(len(param_list)), range(n_folds))))

        # multiprocessing setup
        manager = Manager()
        score_dict = manager.dict()

        def _fit_eval(param_idx, fold_idx, verbose=False, i_rand=-1):
            train_indices, test_indices = train_splits[fold_idx], test_splits[fold_idx]
            X_train, Y_train = X[train_indices], Y[train_indices]
            X_test, Y_test = X[test_indices], Y[test_indices]

            config = tf.ConfigProto(device_count={"CPU": 1},
                                    inter_op_parallelism_threads=1,
                                    intra_op_parallelism_threads=1)

            with tf.Session(config=config):
                kwargs_dict = {**original_params, **param_list[param_idx]}
                kwargs_dict['name'] = 'cv_%i_%i_%i_' % (param_idx, fold_idx, i_rand) + self.name
                model = self.__class__(**kwargs_dict)

                model.fit(X_train, Y_train, verbose=verbose)
                test_score = model.score(X_test, Y_test)
                assert not np.isnan(test_score)
                score_dict[(param_idx, fold_idx)] = test_score

        # run the prepared tasks in multiple processes
        executor = AsyncExecutor(n_jobs=n_jobs)
        executor.run(_fit_eval, param_ids, fold_ids, verbose=verbose)


        # check if all results are available and rerun failed fit_evals. Try three times
        for i in range(3):
            failed_runs = [x for x in zip(param_ids, fold_ids) if x not in score_dict]
            if not failed_runs:
                break
            if verbose:
                print("{} runs succeeded, {} runs failed. Rerunning failed runs".format(len(score_dict.keys()),
                                                                                        len(failed_runs)))
            for (p, f) in failed_runs:
                try:
                    _fit_eval(p, f, verbose=verbose, i_rand=i)
                except Exception as e:
                    print(e)

        # make sure we ultimately have an output for every parameter - fold - combination
        assert len(score_dict.keys()) == len(param_list) * len(train_splits)

        # Select the best parameter setting
        scores_array = np.zeros((len(param_list), len(train_splits)))
        for (i, j), score in score_dict.items():
            scores_array[i, j] = score
        avg_scores = np.mean(scores_array, axis=-1)
        best_idx = np.argmax(avg_scores)
        selected_params = param_list[best_idx]
        assert len(avg_scores) == len(param_list)

        if verbose:
            print("Completed grid search - Selected params: {}".format(selected_params))
            print("Refitting model with selected params")

        # Refit with best parameter set
        self.set_params(**selected_params)
        self.reset_fit()
        self.fit(X, Y, verbose=False)
        return selected_params

    def pdf(self, X, Y):
        """ Predicts the conditional probability p(y|x). Requires the model to be fitted.

           Args:
             X: numpy array to be conditioned on - shape: (n_samples, n_dim_x)
             Y: numpy array of y targets - shape: (n_samples, n_dim_y)

           Returns:
              conditional probability p(y|x) - numpy array of shape (n_query_samples, )

         """
        assert self.fitted, "model must be fitted to compute likelihood score"
        X, Y = self._handle_input_dimensionality(X, Y, fitting=False)
        p = self.sess.run(self.pdf_, feed_dict={self.X_ph: X, self.Y_ph: Y})
        assert p.ndim == 1 and p.shape[0] == X.shape[0]
        return p

    def cdf(self, X, Y):
        """ Predicts the conditional cumulative probability p(Y<=y|X=x). Requires the model to be fitted.

           Args:
             X: numpy array to be conditioned on - shape: (n_samples, n_dim_x)
             Y: numpy array of y targets - shape: (n_samples, n_dim_y)

           Returns:
             conditional cumulative probability p(Y<=y|X=x) - numpy array of shape (n_query_samples, )

        """
        assert self.fitted, "model must be fitted to compute likelihood score"
        X, Y = self._handle_input_dimensionality(X, Y, fitting=False)
        p = self.sess.run(self.cdf_, feed_dict={self.X_ph: X, self.Y_ph: Y})
        assert p.ndim == 1 and p.shape[0] == X.shape[0]
        return p

    def log_pdf(self, X, Y):
        """ Predicts the conditional log-probability log p(y|x). Requires the model to be fitted.

           Args:
             X: numpy array to be conditioned on - shape: (n_samples, n_dim_x)
             Y: numpy array of y targets - shape: (n_samples, n_dim_y)

           Returns:
              onditional log-probability log p(y|x) - numpy array of shape (n_query_samples, )

         """
        assert self.fitted, "model must be fitted to compute likelihood score"
        X, Y = self._handle_input_dimensionality(X, Y, fitting=False)
        p = self.sess.run(self.log_pdf_, feed_dict={self.X_ph: X, self.Y_ph: Y})
        assert p.ndim == 1 and p.shape[0] == X.shape[0]
        return p

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
        sess = tf.get_default_session()
        sess.run([
            tf.assign(self.mean_x_sym, self.x_mean),
            tf.assign(self.std_x_sym, self.x_std),
            tf.assign(self.mean_y_sym, self.y_mean),
            tf.assign(self.std_y_sym, self.y_std)
        ])

    def _compute_noise_intensity(self, X, Y):
        # computes the noise intensity based on the number of samples and dimensionality of the data

        n_samples = X.shape[0]

        if self.adaptive_noise_fn is not None:
            self.x_noise_std = self.adaptive_noise_fn(n_samples, self.ndim_x + self.ndim_y)
            self.y_noise_std = self.adaptive_noise_fn(n_samples, self.ndim_x + self.ndim_y)

            assert self.x_noise_std >= 0.0 and self.y_noise_std >= 0.0

            # assign them to tf variables
            sess = tf.get_default_session()
            sess.run([
                tf.assign(self.x_noise_std_sym, self.x_noise_std),
                tf.assign(self.y_noise_std_sym, self.y_noise_std),
            ])


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

        if self.x_noise_std is None:
            self.x_noise_std = 0.0
        if self.y_noise_std is None:
            self.y_noise_std = 0.0

        # add noise layer if desired
        layer_in_x = L.GaussianNoiseLayer(layer_in_x, self.x_noise_std, noise_on_ph=self.train_phase, name='x')
        self.x_noise_std_sym = layer_in_x.get_params()[0]
        layer_in_y = L.GaussianNoiseLayer(layer_in_y, self.y_noise_std, noise_on_ph=self.train_phase, name='y')
        self.y_noise_std_sym = layer_in_y.get_params()[0]

        # setup dropout. This placeholder will remain unused if dropout is not implemented by the MLP
        self.dropout_ph = tf.placeholder_with_default(0., shape=())

        return layer_in_x, layer_in_y

    def _add_l1_l2_regularization(self, core_network):
        if self.l1_reg > 0 or self.l2_reg > 0:

            # weight norm should not be combined with l1 / l2 regularization
            if self.weight_normalization is True:
                warnings.WarningMessage("l1 / l2 regularization has no effect when weigh normalization is used")

            weight_vector = tf.concat(
                [tf.reshape(param, (-1,)) for param in core_network.get_params_internal() if '/W' in param.name],
                axis=0)
            if self.l2_reg > 0:
                self.l2_reg_loss = self.l2_reg * tf.reduce_sum(weight_vector ** 2)
                tf.losses.add_loss(self.l2_reg_loss, tf.GraphKeys.REGULARIZATION_LOSSES)
            if self.l1_reg > 0:
                self.l1_reg_loss = self.l1_reg * tf.reduce_sum(tf.abs(weight_vector))
                tf.losses.add_loss(self.l1_reg_loss, tf.GraphKeys.REGULARIZATION_LOSSES)

    def __getstate__(self):
        state = LayersPowered.__getstate__(self)
        state['fitted'] = self.fitted
        return state

    def __setstate__(self, state):
        LayersPowered.__setstate__(self, state)
        self.fitted = state['fitted']
        self.sess = tf.get_default_session()

    def _handle_input_dimensionality(self, X, Y=None, fitting=False):
        assert (self.ndim_x == 1 and X.ndim == 1) or (X.ndim == 2 and X.shape[1] == self.ndim_x), "expected X to have shape (?, %i) but received %s"%(self.ndim_x, str(X.shape))
        assert (Y is None) or (self.ndim_y == 1 and Y.ndim == 1) or (Y.ndim == 2 and Y.shape[1] == self.ndim_y), "expected Y to have shape (?, %i) but received %s"%(self.ndim_y, str(Y.shape))
        return BaseDensityEstimator._handle_input_dimensionality(self, X, Y, fitting=fitting)

    @staticmethod
    def _check_uniqueness_of_scope(name):
        current_scope = tf.get_variable_scope().name
        scopes = set([variable.name.split('/')[0] for variable in tf.global_variables(scope=current_scope)])
        assert name not in scopes, "%s is already in use for a tensorflow scope - please choose another estimator name"%name

