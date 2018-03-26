import itertools
import numpy as np
import copy
from cde.tests.ConfigRunner import ConfigRunner
from cde.density_simulation import GaussianMixture, EconDensity
from cde.density_estimator import MixtureDensityNetwork, KernelMixtureNetwork


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
        [0.01, 0.1, None]  # y_noise_std
        )),
  'MixtureDensityNetwork': tuple(itertools.product(
         [5, 10, 20],  # n_centers
         [None],  # estimator
         [None],  # X_ph
         [1000],  # n training epochs
         [0.01, 0.1, None],  # x_noise_std
         [0.01, 0.1, None]  # y_noise_std
  ))}


  simulators_params = {
  'EconDensity': tuple([1]),  # std
  'GaussianMixture': (30, 1, 1, 4.5)  # n_kernels, ndim_x, ndim_y, means_std
  }

  return estimator_params, simulators_params



if __name__ == '__main__':
    keys_of_interest = ['estimator', 'simulator', 'n_observations', 'center_sampling_method', "x_noise_std",
                        'y_noise_std', 'ndim_x', 'ndim_y', 'n_centers', 'kl_divergence',
                    'hellinger_distance', 'js_divergence', 'x_cond', 'random_seed']


    conf_est, conf_sim = issue1()
    conf_runner = ConfigRunner(conf_est, conf_sim, n_observations=100*2**np.arange(0, 7), keys_of_interest=keys_of_interest)
    conf_runner.run_configurations(output_dir="./", prefix_filename="issue1_noise_reg", parallelized=False)
    #import pandas as pd
    #df = pd.read_csv("issue1_noise_reg_result_03-26-18_20-29-41.csv")
    #print(df.head())
