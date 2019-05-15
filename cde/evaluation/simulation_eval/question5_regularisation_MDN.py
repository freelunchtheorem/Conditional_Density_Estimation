import matplotlib as mpl

mpl.use("PS")  # handles X11 server detection (required to run on console)
import numpy as np
from cde.model_fitting.GoodnessOfFitResults import GoodnessOfFitResults
from cde.evaluation.simulation_eval import base_experiment
from ml_logger import logger

EXP_PREFIX = 'question5_regularisation_MDN'
RESULTS_FILE = 'results.pkl'


def question5():
    estimator_params = {
        'MixtureDensityNetwork':
            {
                'n_centers': [10],
                'n_training_epochs': [1000],
                'hidden_sizes': [(16, 16)],
                'x_noise_std': [0.1, 0.2, 0.4, None],
                'y_noise_std': [0.01, 0.1, 0.2, None],
                'dropout': [0., 0.2],
                'weight_decay': [0., 5e-5],
                'weight_normalization': [False, True],
                'random_seed': [22]
            }
    }

    simulators_params = {
        'EconDensity': {
            'std': [1],
            'heteroscedastic': [True],
        },
        'GaussianMixture': {
            'n_kernels': [10],
            'ndim_x': [1],
            'ndim_y': [1],
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

    observations = 100 * np.logspace(0, 6, num=7, base=2.0, dtype=np.int32)

    return estimator_params, simulators_params, observations


if __name__ == '__main__':
    estimator_params, simulators_params, observations = question5()
    load = base_experiment.launch_experiment(estimator_params, simulators_params, observations, EXP_PREFIX)
