import itertools
import pickle
import numpy as np

from cde.evaluation.ConfigRunner import ConfigRunner


def issue1():
  estimator_params = {
  'KernelMixtureNetwork': tuple(itertools.product(
        ["k_means"],  # center_sampling_method
        [5, 10, 50, 100], # n_centers
        [True],  # keep_edges
        [[0.1, 0.5, 1.]],  # init_scales
        [None],  # estimator
        [None],  # X_ph
        [True],  # train_scales
        [1000],  # n training epochs
        [0.01, 0.1, None],  # x_noise_std
        [0.01, 0.1, None],  # y_noise_std
        [22]  # random_seed
        )),
  'MixtureDensityNetwork': tuple(itertools.product(
         [5, 10, 20],  # n_centers
         [None],  # estimator
         [None],  # X_ph
         [1000],  # n training epochs
         [0.01, 0.1, None],  # x_noise_std
         [0.01, 0.1, None],  # y_noise_std
         [22]  # random_seed
  ))}


  simulators_params = {
  'EconDensity': tuple([1]),  # std
  'GaussianMixture': (30, 2, 2, 4.5, 22)  # n_kernels, ndim_x, ndim_y, means_std, #random_seed
  }

  return estimator_params, simulators_params



if __name__ == '__main__':
    keys_of_interest = ['estimator', 'simulator', 'n_observations', 'center_sampling_method', 'x_noise_std',
                        'y_noise_std', 'ndim_x', 'ndim_y', 'n_centers', "n_mc_samples", "n_x_cond", 'mean_est',
                        'cov_est', 'mean_sim', 'cov_sim','kl_divergence', 'hellinger_distance', 'js_divergence',
                        'x_cond', 'random_seed',
                        ]


    conf_est, conf_sim = issue1()
    conf_runner = ConfigRunner(conf_est, conf_sim, n_observations=100*2**np.arange(0, 7), keys_of_interest=keys_of_interest,
                               n_mc_samples=10**3, n_x_cond=5)
    results_list, full_df, path_pickle = conf_runner.run_configurations(limit=2, output_dir="./", prefix_filename="question1_noise_reg")
    with open(path_pickle, 'rb') as pickle_file:
      content = pickle.load(pickle_file)
      print(content)