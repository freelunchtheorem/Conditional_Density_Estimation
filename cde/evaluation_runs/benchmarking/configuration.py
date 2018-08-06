import pickle
import numpy as np
import glob
import os

from cde.evaluation.ConfigRunner import ConfigRunner
from ml_logger import logger

EXP_PREFIX = 'benchmarking'

def question1(): #noise
  estimator_params = {
  'KernelMixtureNetwork':

    {'center_sampling_method': ["k_means"],
     'n_centers': [10],
     'keep_edges': [True],
     'init_scales': [[0.1, 0.5, 1.]],
     'train_scales': [True],
     'hidden_sizes': [(16, 16)],
     'n_training_epochs': [500],
     'x_noise_std': [0.01, None],
     'y_noise_std': [None],
     'random_seed': [22]
     }}

  simulators_params = {
  'EconDensity': {'std': [1],
                  'heteroscedastic': [True],
                  }
  }


  return estimator_params, simulators_params


if __name__ == '__main__':

  run = True
  load = not run

  keys_of_interest = ['task_name', 'estimator', 'simulator', 'n_observations', 'center_sampling_method', 'x_noise_std',
                      'y_noise_std', 'ndim_x', 'ndim_y', 'n_centers', "n_mc_samples", "n_x_cond", 'mean_est',
                      'cov_est', 'mean_sim', 'cov_sim', 'kl_divergence', 'hellinger_distance', 'js_divergence',
                      'x_cond', 'random_seed', "mean_sim", "cov_sim", "mean_abs_diff", "cov_abs_diff",
                      "VaR_sim", "VaR_est", "VaR_abs_diff", "CVaR_sim", "CVaR_est", "CVaR_abs_diff",
                      "time_to_fit"
                      ]


  if run:
      observations = np.asarray([800])

      conf_est, conf_sim = question1()
      conf_runner = ConfigRunner(EXP_PREFIX, conf_est, conf_sim, observations=observations, keys_of_interest=keys_of_interest,
                                 n_mc_samples=2*10**6, n_x_cond=5, n_seeds=5)

      results_list, full_df = conf_runner.run_configurations(dump_models=True, multiprocessing=False, n_workers=2)

  # if load:
  #   with open(results_pickle, 'rb') as pickle_file:
  #     gof_result = pickle.load(pickle_file)
  #     results_df = gof_result.generate_results_dataframe(keys_of_interest)
  #
  #
  #     graph_dicts = [
  #       {"estimator": "KernelMixtureNetwork", "x_noise_std": 0.1, "y_noise_std": None, "n_centers": 20},
  #       {"estimator": "KernelMixtureNetwork", "x_noise_std": 0.01, "y_noise_std": None, "n_centers": 20},
  #       {"estimator": "KernelMixtureNetwork", "x_noise_std": None, "y_noise_std": None, "n_centers": 20},
  #       {"estimator": "MixtureDensityNetwork", "x_noise_std": 0.1, "y_noise_std": None, "n_centers": 10},
  #       { "estimator": "MixtureDensityNetwork", "x_noise_std": 0.01, "y_noise_std": None, "n_centers": 10},
  #       {"estimator": "MixtureDensityNetwork", "x_noise_std": None, "y_noise_std": None, "n_centers": 10}
  #     ]
  #
  #     gof_result.plot_metric(graph_dicts, metric="kl_divergence", simulator="EconDensity")
  #     print(gof_result)