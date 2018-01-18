from joblib import Parallel, delayed
import numpy as np
import itertools
from density_estimator import LSConditionalDensityEstimation, KernelMixtureNetwork
from density_simulation import EconDensity, GaussianMixture
from evaluation.GoodnessOfFit import GoodnessOfFit, GoodnessOfFitResults




def prepare_configurations():

  """ configurations """
  estimator_params = {'KMN': tuple(itertools.product(["agglomerative", "k_means", "random"],  # center_sampling_method
    [10, 20, 50],  # n_centers
    [True, False],  # keep_edges
    [[0.1], [0.2], [0.5], [1], [2], [5]],  # init_scales
    [None], # estimator
    [None], #X_ph
    [True, False],  # train_scales
    [5])), # n training epochs
    'LSCDE': tuple(itertools.product(['k_means'],  # center_sampling_method
      [0.1, 0.2, 1, 2, 5],  # bandwidth
      [10, 20, 50],  # n_centers
      [0.1, 0.2, 0.4, 0.5, 1],  # regularization
      [False, True]))}  # keep_edges}

  simulators_params = {
      'Econ': tuple([0]), # std
      'GMM': (30, 1, 1, 4.5)  #n_kernels, ndim_x, ndim_y, means_std
  }


  """ object references """
  estimator_references = {'KMN': KernelMixtureNetwork, 'LSCDE': LSConditionalDensityEstimation, }
  simulator_references = { 'Econ': EconDensity, 'GMM': GaussianMixture}

  """ estimators """
  configured_estimators = [estimator_references[estimator](*config) for estimator, est_params in estimator_params.items() for config in est_params]

  """ simulators """
  configured_simulators = [simulator_references[simulator](*sim_params) for simulator, sim_params in simulators_params.items()]
  del estimator_references
  del simulator_references

  return configured_estimators, configured_simulators



def run_configurations(configured_estimators, configured_simulators):

  results = Parallel(n_jobs=-1)(delayed(run_single_configuration)(estimator, simulator, n_observations=200) for estimator, simulator in
                           itertools.product(configured_estimators, configured_simulators))

  # todo: check __reduce__ function in KMN.py
  # todo: work with GoodnessOfFitResults


def run_single_configuration(estimator, simulator, n_observations):
    gof = GoodnessOfFit(estimator=estimator, probabilistic_model=simulator, n_observations=n_observations)
    ks, p = gof.kolmogorov_smirnov_2sample()
    kl = gof.kl_divergence()
    return ks, p, kl



def main():
  conf_est, conf_sim = prepare_configurations()
  run_configurations(conf_est, conf_sim)


if __name__ == "__main__": main()
