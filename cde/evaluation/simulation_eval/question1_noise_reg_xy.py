import matplotlib as mpl

mpl.use("PS")  # handles X11 server detection (required to run on console)
import numpy as np
from cde.model_fitting.GoodnessOfFitResults import GoodnessOfFitResults
from cde.evaluation.simulation_eval import base_experiment
from ml_logger import logger

EXP_PREFIX = 'question1_noise_reg_xy_v1'
RESULTS_FILE = 'results.pkl'


def question1():
    estimator_params = {
        'KernelMixtureNetwork':
            {
                'center_sampling_method': ["k_means"],
                'n_centers': [20],
                'keep_edges': [True],
                'init_scales': [[0.1, 0.5, 1.], [0.3, 0.7]],
                'train_scales': [True],
                'hidden_sizes': [(16, 16)],
                'n_training_epochs': [1000],
                'x_noise_std': [0.1, 0.2, 0.4, None],
                'y_noise_std': [0.01, 0.02, 0.05, 0.1, 0.2, None],
                'random_seed': [22],
            },
        'NormalizingFlowEstimator':
            {
                'flows_type': [('affine', 'radial', 'radial', 'radial')],
                'n_training_epochs': [1000],
                'hidden_sizes': [(16, 16)],
                'x_noise_std': [0.1, 0.2, 0.4, None],
                'y_noise_std': [0.01, 0.02, 0.05, 0.1, 0.2, None],
                'random_seed': [22]
            },
        'MixtureDensityNetwork':
            {
                'n_centers': [10],
                'n_training_epochs': [1000],
                'hidden_sizes': [(16, 16)],
                'x_noise_std': [0.1, 0.2, 0.4, None],
                'y_noise_std': [0.01, 0.02, 0.05, 0.1, 0.2, None],
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
    estimator_params, simulators_params, observations = question1()
    load = base_experiment.launch_experiment(estimator_params, simulators_params, observations, EXP_PREFIX)
