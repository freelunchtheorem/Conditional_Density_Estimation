import matplotlib as mpl
mpl.use("PS") #handles X11 server detection (required to run on console)
import numpy as np
from cde.evaluation.GoodnessOfFitResults import GoodnessOfFitResults
from cde.evaluation_runs import base_experiment
import cde.evaluation.ConfigRunner as ConfigRunner
#from cde.evaluation.ConfigRunner import load_dumped_estimators
from ml_logger import logger

EXP_PREFIX = 'question1_noise_reg_x'
RESULTS_FILE = 'results.pkl'

def question1():
  estimator_params = {
  'KernelMixtureNetwork':

    {'center_sampling_method': ["k_means"],
     'n_centers': [20],
     'keep_edges': [True],
     'init_scales': [[0.1, 0.5, 1.]],
     'train_scales': [True],
     'hidden_sizes': [(16, 16)],
     'n_training_epochs': [1000],
     'x_noise_std': [0.1, 0.2, 0.4, None],
     'y_noise_std': [0.1, 0.2, 0.4, None],
     'random_seed': [22],
     },
  'MixtureDensityNetwork':
    {
      'n_centers': [10],
      'n_training_epochs': [1000],
      'hidden_sizes': [(16, 16)],
      'x_noise_std': [0.1, 0.2, 0.4, None],
      'y_noise_std': [0.1, 0.2, 0.4, None],
      'random_seed': [22]
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
    load = base_experiment.launch_experiment(estimator_params, simulators_params, observations, EXP_PREFIX)

    logger.configure('/Users/fabioferreira/Dropbox/0_Studium/Master/git_projects/Nonparametric_Density_Estimation/data/local', EXP_PREFIX)

    if load:
      results_from_pkl_file = dict(logger.load_pkl_log(RESULTS_FILE))
      gof_result = GoodnessOfFitResults(single_results_dict=results_from_pkl_file)
      results_df = gof_result.generate_results_dataframe(base_experiment.KEYS_OF_INTEREST)

      #logger.load_pkl("model_dumps/MixtureDensityNetwork_task_663.pkl")
      gof_result = ConfigRunner.load_dumped_estimators(gof_result, task_id=[5])

      graph_dicts = [
      # {"estimator": "KernelMixtureNetwork", "x_noise_std": 0.2, "y_noise_std": None, "n_centers": 20},
      # {"estimator": "KernelMixtureNetwork", "x_noise_std": 0.1, "y_noise_std": None, "n_centers": 20},
      # {"estimator": "KernelMixtureNetwork", "x_noise_std": 0.2, "y_noise_std": None, "n_centers": 50},
      # {"estimator": "KernelMixtureNetwork", "x_noise_std": 0.1, "y_noise_std": None, "n_centers": 50},
      # {"estimator": "KernelMixtureNetwork", "x_noise_std": 0.01, "y_noise_std": None, "n_centers": 20},
      # {"estimator": "KernelMixtureNetwork", "x_noise_std": None, "y_noise_std": None, "n_centers": 20},
      {"estimator": "MixtureDensityNetwork", "x_noise_std": 0.2, "y_noise_std": None, "n_centers": 10},
      {"estimator": "MixtureDensityNetwork", "x_noise_std": 0.1, "y_noise_std": None, "n_centers": 10},
      # {"estimator": "MixtureDensityNetwork", "x_noise_std": 0.01, "y_noise_std": None, "n_centers": 10},
      # {"estimator": "MixtureDensityNetwork", "x_noise_std": None, "y_noise_std": None, "n_centers": 10}
      {"estimator": "MixtureDensityNetwork", "x_noise_std": 0.2, "y_noise_std": None, "n_centers": 20},
      {"estimator": "MixtureDensityNetwork", "x_noise_std": 0.1, "y_noise_std": None, "n_centers": 20},
      # {"estimator": "MixtureDensityNetwork", "x_noise_std": 0.01, "y_noise_std": None, "n_centers": 20},
      # {"estimator": "MixtureDensityNetwork", "x_noise_std": None, "y_noise_std": None, "n_centers": 20}
      ]

      gof_result.plot_metric(graph_dicts, metric="js_divergence", simulator="EconDensity")
      print(results_df)

