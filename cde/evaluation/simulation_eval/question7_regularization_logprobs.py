import matplotlib as mpl

mpl.use("PS")  # handles X11 server detection (required to run on console)
import numpy as np
import types
from cde.model_fitting.GoodnessOfFitResults import GoodnessOfFitResults
from cde.evaluation.simulation_eval import base_experiment
from ml_logger import logger

EXP_PREFIX = 'question7_regularization_logprob'
RESULTS_FILE = 'results.pkl'


def question7():

    class Rule_of_thumb:

        def __init__(self, scale_factor):
            self.scale_factor = scale_factor

        def __call__(self, n, d):
            return self.scale_factor * n**(-1 / (4+d))

        def __str__(self):
            return "rule_of_thumb_%.2f" % self.scale_factor

    class Quadratic_Rate:
        def __init__(self, scale_factor):
            self.scale_factor = scale_factor

        def __call__(self, n, d):
            return self.scale_factor * n**(-1 / (1+d))

        def __str__(self):
            return "quadratic_rate_%.2f" % self.scale_factor

    class Polynomial_Rate:
        def __init__(self, scale_factor, order):
            self.scale_factor = scale_factor
            self.order = order

        def __call__(self, n, d):
            return self.scale_factor * n ** (-1 / (self.order + d))

        def __str__(self):
            return "polynomial_rate_%i_%.2f" % (self.order, self.scale_factor)

    adaptive_noise_functions = [Rule_of_thumb(0.0), Rule_of_thumb(1.0), Rule_of_thumb(0.7), Rule_of_thumb(0.5),
                                Quadratic_Rate(2.0), Quadratic_Rate(1.0), Polynomial_Rate(1.0, 2), Polynomial_Rate(2.0, 2),
                                Polynomial_Rate(1.0, 3), Polynomial_Rate(2.0, 3)]
    l1_reg_params = [0.0, 0.001, 0.01, 0.1, 1.0]
    l2_reg_params = [0.0, 0.001, 0.01, 0.1, 1.0]
    weight_decay_params = [0.0, 0.001, 0.01, 0.1]


    estimator_params = [

        # MDNs  ##############

        ('MixtureDensityNetwork', # noise regularization
            {
                'n_centers': [10],
                'n_training_epochs': [1000],
                'hidden_sizes': [(32, 32)],
                'x_noise_std': [None],
                'y_noise_std': [None],
                'adaptive_noise_fn': adaptive_noise_functions,
                'weight_decay': [0.],
                'weight_normalization': [False],
                'random_seed': [22]
            }
        ),

        ('MixtureDensityNetwork',
         {
             'n_centers': [10],
             'n_training_epochs': [1000],
             'hidden_sizes': [(32, 32)],
             'x_noise_std': [None],
             'y_noise_std': [None],
             'adaptive_noise_fn': [None],
             'weight_decay': weight_decay_params,
             'weight_normalization': [False],
             'random_seed': [22]
         }
         ),

        ('MixtureDensityNetwork',
         {
             'n_centers': [10],
             'n_training_epochs': [1000],
             'hidden_sizes': [(32, 32)],
             'x_noise_std': [None],
             'y_noise_std': [None],
             'adaptive_noise_fn': [None],
             'weight_decay': [0.0],
             'l2_reg': l2_reg_params,
             'weight_normalization': [False],
             'random_seed': [22]
         }
         ),

        ('MixtureDensityNetwork',
         {
             'n_centers': [10],
             'n_training_epochs': [1000],
             'hidden_sizes': [(32, 32)],
             'x_noise_std': [None],
             'y_noise_std': [None],
             'adaptive_noise_fn': [None],
             'l1_reg': l1_reg_params,
             'weight_normalization': [False],
             'random_seed': [22]
         }
         ),

        # KMNs  ##############

        ('KernelMixtureNetwork',
         {
             'n_centers': [50],
             'n_training_epochs': [1000],
             'hidden_sizes': [(32, 32)],
             'x_noise_std': [None],
             'y_noise_std': [None],
             'adaptive_noise_fn': adaptive_noise_functions,
             'weight_decay': [0.0],
             'weight_normalization': [True, False],
             'random_seed': [22]
         }),

        ('KernelMixtureNetwork',
            {
                'n_centers': [50],
                'n_training_epochs': [1000],
                'hidden_sizes': [(32, 32)],
                'x_noise_std': [None],
                'y_noise_std': [None],
                'adaptive_noise_fn': [None],
                'weight_decay': weight_decay_params,
                'weight_normalization': [False],
                'random_seed': [22]
            }),
        ('KernelMixtureNetwork',
         {
             'n_centers': [50],
             'n_training_epochs': [1000],
             'hidden_sizes': [(32, 32)],
             'x_noise_std': [None],
             'y_noise_std': [None],
             'adaptive_noise_fn': [None],
             'l2_reg': l2_reg_params,
             'weight_normalization': [False],
             'random_seed': [22]
         }),
        ('KernelMixtureNetwork',
         {
             'n_centers': [50],
             'n_training_epochs': [1000],
             'hidden_sizes': [(32, 32)],
             'x_noise_std': [None],
             'y_noise_std': [None],
             'adaptive_noise_fn': [None],
             'l1_reg': l1_reg_params,
             'weight_normalization': [False],
             'random_seed': [22]
         }),

        # NFs ##################################

        ('NormalizingFlowEstimator',
            {
                'n_flows': [10],
                'n_training_epochs': [1000],
                'hidden_sizes': [(32, 32)],
                'x_noise_std': [None],
                'y_noise_std': [None],
                'adaptive_noise_fn': adaptive_noise_functions,
                'weight_decay': [0.],
                'weight_normalization': [True, False],
                'random_seed': [22]
            }),

        ('NormalizingFlowEstimator',
         {
             'n_flows': [10],
             'n_training_epochs': [1000],
             'hidden_sizes': [(32, 32)],
             'x_noise_std': [None],
             'y_noise_std': [None],
             'adaptive_noise_fn': [None],
             'weight_decay': weight_decay_params,
             'weight_normalization': [False],
             'random_seed': [22]
         }),

        ('NormalizingFlowEstimator',
         {
             'n_flows': [10],
             'n_training_epochs': [1000],
             'hidden_sizes': [(32, 32)],
             'x_noise_std': [None],
             'y_noise_std': [None],
             'adaptive_noise_fn': [None],
             'l2_reg': l2_reg_params,
             'weight_normalization': [False],
             'random_seed': [22]
         }),

        ('NormalizingFlowEstimator',
         {
             'n_flows': [10],
             'n_training_epochs': [1000],
             'hidden_sizes': [(32, 32)],
             'x_noise_std': [None],
             'y_noise_std': [None],
             'adaptive_noise_fn': [None],
             'l1_reg': l1_reg_params,
             'weight_normalization': [False],
             'random_seed': [22]
         }),
    ]

    simulators_params = [
        ('EconDensity',{
            'std': [1],
            'heteroscedastic': [True],
        }),
        ('GaussianMixture', {
            'n_kernels': [5],
            'ndim_x': [2],
            'ndim_y': [2],
            'means_std': [1.5]
        }),
        ('GaussianMixture', {
            'n_kernels': [5],
            'ndim_x': [4],
            'ndim_y': [4],
            'means_std': [1.5]
        }),
        ('SkewNormal', {})
    ]

    observations = 100 * np.logspace(0, 7, num=8, base=2.0, dtype=np.int32)

    return estimator_params, simulators_params, observations


if __name__ == '__main__':
    estimator_params, simulators_params, observations = question7()
    load = base_experiment.launch_logprob_experiment(estimator_params, simulators_params, observations, EXP_PREFIX, 
                                                     n_seeds=5, n_test_samples=2*10**5)
