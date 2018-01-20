from contextlib import contextmanager
import itertools
from density_estimator import LSConditionalDensityEstimation, KernelMixtureNetwork
from density_simulation import EconDensity, GaussianMixture
from evaluation.GoodnessOfFit import GoodnessOfFit
from multiprocessing import Pool, freeze_support
import multiprocessing
import pandas as pd
from utils import io


def prepare_configurations():

  """ configurations """
  estimator_params = {'KMN': tuple(itertools.product(["agglomerative", "k_means", "random"],  # center_sampling_method
    [10, 20, 50],  # n_centers
    [True, False],  # keep_edges
    [[0.1], [0.2], [0.5], [1], [2], [5]],  # init_scales
    [None], # estimator
    [None], #X_ph
    [True, False],  # train_scales
    [500])), # n training epochs
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



def create_configurations(configured_estimators, configured_simulators):
  """
  creates all possible combinations from the (configured) estimators and simulators.
  :param configured_estimators: a list instantiated estimator objects with length n while n being the number of configured estimators
  :param configured_simulators: a list instantiated simulator objects with length n while m being the number of configured simulators
  :return: a list containing n*m=k tuples while k being the number of the cartesian product of estimators and simulators,
  shape of tuples: (estimator object, simulator object)
  """
  return [(estimator, simulator, 2000) for estimator, simulator in itertools.product(configured_estimators, configured_simulators)]


def run_configurations(tasks, estimator_filter=None, parallelized=False, limit=None):
  """
  Runs the given configurations, i.e.
  1) fits the estimator to the simulation and
  2) executes goodness-of-fit (currently: kolmogorov-smirnof (cdf-based), kl divergence) tests
  Every successful run yields a result object of type GoodnessOfFitResult which contains the following members: cond_values, time_to_fit,
  time_to_predict ndim_x, ndim_y, estimator_params, probabilistic_model_params, mean_kl, mean_ks_stat, mean_ks_pval
  :param tasks: a list containing k tuples, each tuple has the shape (estimator object, simulator object)
  :param estimator_filter: a parameter to decide whether to execute just a specific type of estimator, e.g. "KernelMixtureNetwork",
  must be one of the density estimator class types
  :param limit: limit the number of (potentially filtered) tasks
  :param parallelized: if True, the configurations are run in parallel mode on all available cpu's
  :return: a list of GoodnessOfFitResults objects (one per configuration run) and a list of the GoodnessOfFit objects (one per configuration run)
  which contain the fitted estimators
  """
  assert len(tasks) > 0
  if estimator_filter is not None:
    tasks = [tupl for tupl in tasks if estimator_filter in tasks[0][0].__class__.__name__]
  assert len(tasks), "no tasks to execute after filtering for the estimator"
  print("Running configurations. Number of total tasks: ", len(tasks))

  if limit is not None:
    assert limit > 0, "limit mustn't be negative"
  else:
    limit = len(tasks)


  gof_results = []
  gof_objects = []

  if parallelized:
    with poolcontext(processes=None) as pool:
      gof_objects, gof_result = pool.starmap(run_single_configuration, tasks[:limit])

  else:
    for task in tasks[:limit]:
      try:
        gof_object, gof_result = run_single_configuration(*task)
        gof_results.append(gof_result)
        gof_objects.append(gof_object)
      except Exception as e:
        print("error for configuration: ", task)
        print(str(e))

  return gof_objects, gof_results


def run_single_configuration(estimator, simulator, n_observations):
  print(estimator, simulator)
  gof = GoodnessOfFit(estimator=estimator, probabilistic_model=simulator, n_observations=n_observations)
  return gof, gof.compute_results()


def export_results(results, output_dir=None, file_name=None, export_pickle=False, export_csv=False):
  assert len(results) > 0, "no results given"

  columns = ['estimator', 'simulator', 'n_observations', 'ndim_x', 'ndim_y', 'n_centers', 'mean_ks_stat', 'mean_ks_pval', 'mean_kl', 'time_to_fit',
             'time_to_predict']

  result_dicts = [result  .report_dict() for result in results]
  df = pd.DataFrame(result_dicts).reindex(columns, axis=1)

  if export_pickle:
    io.store_dataframe(df, output_dir, file_name)
  if export_csv:
    io.store_csv(df, output_dir, file_name)


def merge_names(a, b):
    return '{} & {}'.format(a, b)

def merge_names_unpack(args):
    return merge_names(*args)

@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()


def main():
  conf_est, conf_sim = prepare_configurations()
  tasks = create_configurations(conf_est, conf_sim)
  gofs, gof_results = run_configurations(tasks, parallelized=False)
  export_results(gof_results, output_dir="./", file_name="evaluated_configs_df_", export_pickle=True, export_csv=True)
  io.store_objects(gofs, output_dir="./", file_name="goodness_of_fit_config_objects_")


if __name__ == "__main__": main()
