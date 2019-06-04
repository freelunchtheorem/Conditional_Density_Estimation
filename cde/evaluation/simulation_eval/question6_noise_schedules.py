import matplotlib as mpl

mpl.use("PS")  # handles X11 server detection (required to run on console)
import numpy as np
import types
from cde.model_fitting.GoodnessOfFitResults import GoodnessOfFitResults
from cde.evaluation.simulation_eval import base_experiment
from ml_logger import logger

EXP_PREFIX = 'question6_noise_schedules'
RESULTS_FILE = 'results.pkl'


def question6():

    class Rule_of_thumb1:
        def __call__(self, n, d):
            return 0.5 * n**(-1 / (4+d))

        def __str__(self):
            return "rule_of_thumb_1"

    class Rule_of_thumb2:
        def __call__(self, n, d):
            return n ** (-1 / (4 + d))

        def __str__(self):
            return "rule_of_thumb_2"

    class Fixed_Rate:
        def __call__(self, n, d):
            return 0.2

        def __str__(self):
            return "fixed_rate_0.2"

    class Quadratic_Rate:
        def __call__(self, n, d):
            return n**(-1 / (1+d))

        def __str__(self):
            return "quadratic_rate"

    estimator_params = {
        'MixtureDensityNetwork':
            {
                'n_centers': [10],
                'n_training_epochs': [1000],
                'hidden_sizes': [(32, 32)],
                'x_noise_std': [None],
                'y_noise_std': [None],
                'adaptive_noise_fn': [Rule_of_thumb1(), Rule_of_thumb2(), Fixed_Rate(), Quadratic_Rate()],
                'dropout': [0.],
                'weight_decay': [0.],
                'weight_normalization': [True],
                'random_seed': [22]
            },
        'KernelMixtureNetwork':
            {
                'n_centers': [50],
                'n_training_epochs': [1000],
                'hidden_sizes': [(32, 32)],
                'x_noise_std': [None],
                'y_noise_std': [None],
                'adaptive_noise_fn': [Rule_of_thumb1(), Rule_of_thumb2(), Fixed_Rate(), Quadratic_Rate()],
                'dropout': [0.],
                'weight_decay': [0.],
                'weight_normalization': [True],
                'random_seed': [22]
            },
        'NormalizingFlowEstimator':
            {
                'flows_type': [('affine', 'radial', 'radial', 'radial')],
                'n_training_epochs': [1000],
                'hidden_sizes': [(32, 32)],
                'x_noise_std': [None],
                'y_noise_std': [None],
                'adaptive_noise_fn': [Rule_of_thumb1(), Rule_of_thumb2(), Fixed_Rate(), Quadratic_Rate()],
                'dropout': [0.],
                'weight_decay': [0.],
                'weight_normalization': [True],
                'random_seed': [22]
            },
    }

    simulators_params = {
        'EconDensity': {
            'std': [1],
            'heteroscedastic': [True],
        },
        'GaussianMixture': {
            'n_kernels': [5],
            'ndim_x': [2],
            'ndim_y': [2],
            'means_std': [1.5]
        },
        'ArmaJump': {
            'c': [0.1],
            'arma_a1': [0.9],
            'std': [0.05],
            'jump_prob': [0.05],
        },
        'SkewNormal': {}
    }

    observations = 100 * np.logspace(0, 7, num=8, base=2.0, dtype=np.int32)

    return estimator_params, simulators_params, observations


if __name__ == '__main__':
    estimator_params, simulators_params, observations = question6()
    load = base_experiment.launch_logprob_experiment(estimator_params, simulators_params, observations, EXP_PREFIX, 
                                                     n_seeds=5, n_test_samples=5*10**5)
