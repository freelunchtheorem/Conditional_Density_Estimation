import itertools
import pickle
import numpy as np

from cde.evaluation.ConfigRunner import ConfigRunner

def issue2():
  estimator_params = {
  'KernelMixtureNetwork':

    {'center_sampling_method': ["k_means"],
     'n_centers': [20],
     'keep_edges': [True],
     'init_scales': [[0.1, 0.5, 1.]],
     'estimator': [None],
     'X_ph': [None],
     'train_scales': [True],
     'n_training_epochs': [100],
     'x_noise_std': [0.01, None],
     'y_noise_std': [0.01, None],
     'random_seed': [22]
     }
  }

  simulators_params = {
  'EconDensity': {'std': [1],
                  'heteroscedastic': [True]
                  },

  'GaussianMixture': {'n_kernels' : [20],
                      'ndim_x': [2],
                      'ndim_y': [2],
                      'means_std': [1.5],
                      'random_seed': [24]
                      }
  }

  return estimator_params, simulators_params


if __name__ == '__main__':

  run = True
  load = not run

  keys_of_interest = ['estimator', 'simulator', 'n_observations', 'center_sampling_method', 'x_noise_std',
                      'y_noise_std', 'ndim_x', 'ndim_y', 'n_centers', "n_mc_samples", "n_x_cond", 'mean_est',
                      'cov_est', 'mean_sim', 'cov_sim', 'kl_divergence', 'hellinger_distance', 'js_divergence',
                      'x_cond', 'random_seed', "mean_sim", "cov_sim"
                      ]

  if run:
    observations = 100 * np.logspace(0, 4, num=5, base=2.0, dtype=np.int32) # creates a list with log scale: 100, 200, 400, 800, 1600

    conf_est, conf_sim = issue2()
    conf_runner = ConfigRunner(conf_est, conf_sim, observations=observations, keys_of_interest=keys_of_interest,
                               n_mc_samples=10**7, n_x_cond=5)

    results_list, full_df = conf_runner.run_configurations(output_dir="./", prefix_filename="question2_NNvsKDE")

  if load:
    path_pickle = "question2_NNvsKDE_result_03-29-18_18-04-22.pickle"
    with open(path_pickle, 'rb') as pickle_file:
      gof_result = pickle.load(pickle_file)
      results_df = gof_result.generate_results_dataframe(keys_of_interest)


      graph_dicts = [
        { "estimator": "MixtureDensityNetwork", "x_noise_std": 0.01, "y_noise_std": 0.01},
        { "estimator": "MixtureDensityNetwork", "x_noise_std": 0.1, "y_noise_std": 0.01},
        {"estimator": "MixtureDensityNetwork", "x_noise_std": None, "y_noise_std": 0.01}
      ]

      gof_result.plot_metric(graph_dicts)
      print(gof_result)