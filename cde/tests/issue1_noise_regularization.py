import itertools
import numpy as np
from cde.tests.evaluate_configurations import create_configurations, run_configurations


def issue1():
  estimator_params = {
  'KMN': tuple(itertools.product(["agglomerative", "k_means", "random"],  # center_sampling_method
        [20, 50], # n_centers
        [True],  # keep_edges
        [[0.1], [1]],  # init_scales
        [None],  # estimator
        [None],  # X_ph
        [True],  # train_scales
        [1000],  # n training epochs
        [0.1, 0.5, 1., 2.],  # x_noise_std
        [0.1, 0.5, 1., 2.]  # y_noise_std
        )),
  'MDN': tuple(itertools.product(["agglomerative", "k_means", "random"],  # center_sampling_method
         [20, 50],  # n_centers
         [None],  # estimator
         [None],  # X_ph
         [1000],  # n training epochs
         [0.1, 0.5, 1., 2.],  # x_noise_std
         [0.1, 0.5, 1., 2.]  # y_noise_std
  ))}

  simulators_params = {
  'Econ': tuple([0]),  # std
  'GMM': (30, 1, 1, 4.5)  # n_kernels, ndim_x, ndim_y, means_std
  }


  return create_configurations(estimator_params, simulators_params, np.arange(100, 10100, 100))


if __name__ == '__main__':
    configs = issue1()
    run_configurations(configs, output_dir="./", prefix_filename="issue1_noise_reg", parallelized=False)