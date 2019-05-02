import matplotlib as mpl
import numpy as np

mpl.use("PS")  # handles X11 server detection (required to run on console)

from cde.evaluation.simulation_eval import base_experiment

EXP_PREFIX = 'hyperparam_sweep'
N_MC_SAMPLES = int(10 ** 5)


def question1():
    estimator_params = {
        'KernelMixtureNetwork':

            {'center_sampling_method': ["k_means"],
             'n_centers': [50, 200],
             'keep_edges': [True],
             'init_scales': [[0.2, 0.5, 0.8], [0.3, 0.7]],
             'train_scales': [True],
             'hidden_sizes': [(16, 16)],
             'n_training_epochs': [2000],
             'x_noise_std': [0.1],
             'y_noise_std': [0.1],
             'weight_normalization': [True],
             'data_normalization': [True, False]
             },
        'NormalizingFlowEstimator':
            {
                'flows_type': [('affine', 'radial', 'radial', 'radial', 'radial')],
                'n_training_epochs': [1000, 2000],
                'hidden_sizes': [(16, 16)],
                'x_noise_std': [0.1],
                'y_noise_std': [0.1],
                'weight_normalization': [True],
                'data_normalization': [True, False],
            },
        'MixtureDensityNetwork':
            {
                'n_centers': [10, 20],
                'n_training_epochs': [2000],
                'hidden_sizes': [(16, 16), (32, 32)],
                'x_noise_std': [0.1],
                'y_noise_std': [0.1],
                'weight_normalization': [True],
                'data_normalization': [True, False]

            }
    }

    simulators_params = {
        'EconDensity': {'std': [1],
                        'heteroscedastic': [True],
                        },
        'ArmaJump': {},
        'SkewNormal': {}
    }

    observations = 100 * np.logspace(2, 6, num=7, base=2.0, dtype=np.int32)

    return estimator_params, simulators_params, observations


if __name__ == '__main__':
    estimator_params, simulators_params, observations = question1()
    base_experiment.launch_experiment(estimator_params, simulators_params, observations, EXP_PREFIX,
                                      n_mc_samples=N_MC_SAMPLES, tail_measures=False)
