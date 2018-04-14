import pickle
import numpy as np
import glob
from cde.evaluation.ConfigRunner import ConfigRunner


def question2(): #noise
  estimator_params = {
  'MixtureDensityNetwork':
    {
      'n_centers': [10, 20],
      'estimator': [None],
      'X_ph': [None],
      'n_training_epochs': [2000],
      'x_noise_std': [None],
      'y_noise_std': [0.1, 0.2, 0.5, None],
      'random_seed': [22]
    }
  }

  simulators_params = {
  'EconDensity': {'std': [1],
                  'heteroscedastic': [True],
                  },
  }


  return estimator_params, simulators_params


if __name__ == '__main__':

  run = True
  load = not run

  keys_of_interest = ['estimator', 'simulator', 'n_observations', 'center_sampling_method', 'x_noise_std',
                      'y_noise_std', 'ndim_x', 'ndim_y', 'n_centers', "n_mc_samples", "n_x_cond", 'mean_est',
                      'cov_est', 'mean_sim', 'cov_sim', 'kl_divergence', 'hellinger_distance', 'js_divergence',
                      'x_cond', 'random_seed', "mean_sim", "cov_sim", "mean_abs_diff", "cov_abs_diff",
                      #"VaR_sim", "VaR_est", "VaR_abs_diff", "CVaR_sim", "CVaR_est", "CVaR_abs_diff",
                      "time_to_fit"
                      ]

  # Search for pickle file in directory
  results_pickle_file = glob.glob("*noise*.pickle")
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
    observations = 100 * np.logspace(0, 4, num=5, base=2.0, dtype=np.int32) # creates a list with log scale: 100, 200, 400, 800, 1600

    conf_est, conf_sim = question2()
    conf_runner = ConfigRunner(conf_est, conf_sim, observations=observations, keys_of_interest=keys_of_interest,
                               n_mc_samples=2*10**6, n_x_cond=5, n_seeds=5, results_pickle_file=results_pickle, config_pickle_file=config_pickle_file)

    results_list, full_df = conf_runner.run_configurations(output_dir="./", prefix_filename="question2_noise_reg")

  if load:

    with open(results_pickle, 'rb') as pickle_file:
      gof_result = pickle.load(pickle_file)
      results_df = gof_result.generate_results_dataframe(keys_of_interest)


      graph_dicts = [
        {"estimator": "MixtureDensityNetwork", "simulator": "EconDensity", "x_noise_std": None, "y_noise_std": 0.1},
        {"estimator": "MixtureDensityNetwork", "simulator": "EconDensity", "x_noise_std": None, "y_noise_std": 0.01},
        {"estimator": "MixtureDensityNetwork", "simulator": "EconDensity", "x_noise_std": None, "y_noise_std": 0.05},
        {"estimator": "MixtureDensityNetwork", "simulator": "EconDensity", "x_noise_std": None, "y_noise_std": None},
      ]

      gof_result.plot_metric(graph_dicts, metric="js_divergence")
      print(gof_result)