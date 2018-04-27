import pickle
import numpy as np
import glob

from cde.evaluation.ConfigRunner import ConfigRunner
from collections import OrderedDict

def question3():
  estimator_params = {

    'ConditionalKernelDensityEstimation':
      {
          'bandwidth_selection': ['normal_reference', 'cv_ml'],
          'random_seed': [22]
      },
   'KernelMixtureNetwork':
  
     {'center_sampling_method': ["k_means"],
      'n_centers': [50],
      'keep_edges': [True],
      'init_scales': [[0.1, 0.5, 1.]],
      'estimator': [None],
      'X_ph': [None],
      'train_scales': [True],
      'n_training_epochs': [800],
      'x_noise_std': [None],
      'y_noise_std': [0.01, 0.1],
      'random_seed': [22],
      },

   'MixtureDensityNetwork':
       {
         'n_centers': [50],
         'estimator': [None],
         'X_ph': [None],
         'n_training_epochs': [800],
         'x_noise_std': [None],
         'y_noise_std': [0.01, 0.1],
         'random_seed': [22]
       },
  }

  simulators_params = {
  'EconDensity': {'std': [1],
                  'heteroscedastic': [True]
                  },

  'ArmaJump': {'c': [0.1],
               'arma_a1': [0.9],
               'std': [0.05],
               'jump_prob': [0.05]
               },

  'GaussianMixture': {'n_kernels': [20],
                      'ndim_x': [2],
                      'ndim_y': [2],
                      'means_std': [1.5]
                },
  }

  return estimator_params, simulators_params


if __name__ == '__main__':

  run = True
  load = not run

  keys_of_interest = ['estimator', 'simulator', 'n_observations', 'center_sampling_method', 'x_noise_std',
                      'y_noise_std', 'ndim_x', 'ndim_y', 'n_centers', "bandwidth_selection", "n_mc_samples", "n_x_cond", 'mean_est',
                      'cov_est', 'mean_sim', 'cov_sim', 'kl_divergence', 'hellinger_distance', 'js_divergence',
                      'x_cond', 'random_seed', "mean_sim", "cov_sim", "mean_abs_diff", "cov_abs_diff", "time_to_fit"
                      ]


  # Search for pickle file in directory
  results_pickle_file = glob.glob("*question3_NNvsKDE*.pickle")
  config_pickle_file = glob.glob("config*.pickle")

  if results_pickle_file:
    results_pickle = results_pickle_file[0]
  else:
    results_pickle = None

  if config_pickle_file:
    config_pickle_file = config_pickle_file[0]
  else:
    config_pickle_file = None


  if run:
    observations = 100 * np.logspace(0, 6, num=5, base=2.0, dtype=np.int32)  # creates a list with log scale: 100, 200, 400, 800, 1600

    conf_est, conf_sim = question3()
    conf_runner = ConfigRunner(conf_est, conf_sim, observations=observations, keys_of_interest=keys_of_interest,
                               n_mc_samples=2*10**6, n_x_cond=5, n_seeds=5, results_pickle_file=results_pickle, config_pickle_file=config_pickle_file)

    results_list, full_df = conf_runner.run_configurations(output_dir="./", prefix_filename="question3_NNvsKDE")

  if load:
    path_pickle = ""
    with open(path_pickle, 'rb') as pickle_file:
      gof_result = pickle.load(pickle_file)
      results_df = gof_result.generate_results_dataframe(keys_of_interest)


      graph_dicts = [
        {"estimator": "KernelMixtureNetwork", "y_noise_std": 0.01},
        {"estimator": "KernelMixtureNetwork", "y_noise_std": 0.01, "n_centers": 50},
        {"estimator": "KernelMixtureNetwork"},
        {"estimator": "MixtureDensityNetwork", "n_centers": 10},
        {"estimator": "MixtureDensityNetwork", "n_centers": 10, "y_noise_std": 0.01},
        {"estimator": "MixtureDensityNetwork", "n_centers": 50},
        {"estimator": "MixtureDensityNetwork", "n_centers": 50, "y_noise_std": 0.01},
        {"estimator": "ConditionalKernelDensityEstimation"},

      ]

      gof_result.plot_metric(graph_dicts, title="NN vs KDE", metric="hellinger_distance", simulator="EconDensity")
      print(gof_result)
