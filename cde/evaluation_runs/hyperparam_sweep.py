import matplotlib as mpl
import numpy as np
mpl.use("PS") #handles X11 server detection (required to run on console)

from cde.evaluation_runs import base_experiment

EXP_PREFIX = 'hyperparam_sweep'

def question1():
  estimator_params = {
  'KernelMixtureNetwork':

    {'center_sampling_method': ["k_means", "all"],
     'n_centers': [20, 50],
     'keep_edges': [True],
     'init_scales': [[0.1, 0.5, 1.]],
     'train_scales': [True, False],
     'hidden_sizes': [(16, 16)],
     'n_training_epochs': [1000],
     'x_noise_std': [None],
     'y_noise_std': [None],
     'random_seed': [22],
     'weight_normalization': [True],
     'data_normalization': [True, False]
     },
  'MixtureDensityNetwork':
    {
      'n_centers': [10, 20],
      'n_training_epochs': [1000],
      'hidden_sizes': [(16, 16)],
      'x_noise_std': [None],
      'y_noise_std': [None],
      'random_seed': [22],
      'weight_normalization': [True],
      'data_normalization': [True, False]

    }
  }

  simulators_params = {
  'EconDensity': {'std': [1],
                  'heteroscedastic': [True],
                  },

  'GaussianMixture': {'n_kernels' : [10],
                      'ndim_x': [2],
                      'ndim_y': [2],
                      'means_std': [1.5]
                      },
  'ArmaJump': {'c': [0.1],
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
    base_experiment.launch_experiment(estimator_params, simulators_params, observations, EXP_PREFIX)


