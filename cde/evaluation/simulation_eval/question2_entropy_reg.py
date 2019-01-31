import matplotlib as mpl
mpl.use("PS") #handles X11 server detection (required to run on console)
import numpy as np
from cde.model_fitting.GoodnessOfFitResults import GoodnessOfFitResults
from cde.evaluation.simulation_eval import base_experiment
from ml_logger import logger

EXP_PREFIX = 'question2_entropy_reg'
RESULTS_FILE = 'results.pkl'

def question2():
  estimator_params = {
  'KernelMixtureNetwork':

    {'center_sampling_method': ["k_means"],
     'n_centers': [20, 50],
     'keep_edges': [True],
     'init_scales': [[0.1, 0.5, 1.]],
     'train_scales': [True],
     'hidden_sizes': [(16, 16)],
     'n_training_epochs': [1000],
     'x_noise_std': [0.2],
     'y_noise_std': [0.2],
     'random_seed': [22],
     'entropy_reg_coef': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
     },

  'MixtureDensityNetwork':
    {
      'n_centers': [10, 20],
      'n_training_epochs': [1000],
      'hidden_sizes': [(16, 16)],
      'x_noise_std': [0.2],
      'y_noise_std': [0.2],
      'random_seed': [22],
      'entropy_reg_coef': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
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
    estimator_params, simulators_params, observations = question2()
    load = base_experiment.launch_experiment(estimator_params, simulators_params, observations, EXP_PREFIX)

    if load:
        results_from_pkl_file = dict(logger.load_pkl_log(RESULTS_FILE))
        gof_result = GoodnessOfFitResults(single_results_dict=results_from_pkl_file)
        results_df = gof_result.generate_results_dataframe(base_experiment.KEYS_OF_INTEREST)

        graph_dicts = [
        {"estimator": "KernelMixtureNetwork", "entropy_reg_coef": 0.001, "n_centers": 20},
        {"estimator": "KernelMixtureNetwork", "entropy_reg_coef":  0.01, "n_centers": 20},
        {"estimator": "KernelMixtureNetwork", "entropy_reg_coef": 0.1, "n_centers": 20},
        {"estimator": "KernelMixtureNetwork", "entropy_reg_coef": 1, "n_centers": 20},
        {"estimator": "KernelMixtureNetwork", "entropy_reg_coef": 10, "n_centers": 20},
        {"estimator": "KernelMixtureNetwork", "entropy_reg_coef": 100, "n_centers": 20}
        ]

        gof_result.plot_metric(graph_dicts, metric="js_divergence")
        print(results_df)

