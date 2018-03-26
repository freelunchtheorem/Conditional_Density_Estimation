import itertools
import multiprocessing
import pandas as pd
import numpy as np
import gc
import copy
import traceback
import logging

from contextlib import contextmanager
from cde.density_estimator import LSConditionalDensityEstimation, KernelMixtureNetwork, MixtureDensityNetwork
from cde.density_simulation import EconDensity, GaussianMixture
from cde.evaluation.GoodnessOfFit import GoodnessOfFit
from cde.utils import io
from multiprocessing import Pool




class ConfigRunner():
  def __init__(self, est_params, sim_params, n_observations):
    assert est_params
    assert sim_params
    assert n_observations.all()

    logging.log(logging.INFO, "creating configurations...")
    self.n_observations = n_observations
    self.configured_estimators = [globals()[estimator_name](*config) for estimator_name, estimator_params in est_params.items() for config in estimator_params]
    self.configured_simulators = [globals()[simulator_name](*sim_params) for simulator_name, sim_params in sim_params.items()]

    self.configs = self._create_configurations()


  def _create_configurations(self):
    """
    creates all possible combinations from the (configured) estimators and simulators.
      Args:
        configured_estimators: a list instantiated estimator objects with length n while n being the number of configured estimators
        configured_simulators: a list instantiated simulator objects with length n while m being the number of configured simulators
        n_observations: either a list or a scalar value that defines the number of observations from the simulation model that are used to train the estimators

      Returns:
        if n_observations is not a list, a list containing n*m=k tuples while k being the number of the cartesian product of estimators and simulators is
        returned --> shape of tuples: (estimator object, simulator object)
        if n_observations is a list, n*m*o=k while o is the number of elements in n_observatons list
    """
    if not np.isscalar(self.n_observations):
      print("total number of configurations to be generated: " + str(len(self.configured_estimators) * len(self.configured_simulators) * len(
        self.n_observations)))
      return [copy.deepcopy((estimator, simulator, n_obs)) for estimator, simulator, n_obs in itertools.product(self.configured_estimators,
                                                                                                self.configured_simulators, self.n_observations)]
    else:
      print("total number of configurations to be generated: " + str(len(self.configured_estimators) * len(self.configured_simulators) * self.n_observations))
      return [copy.deepcopy((estimator, simulator, self.n_observations)) for estimator, simulator in itertools.product(self.configured_estimators,
                                                                                                                  self.configured_simulators)]

  def run_configurations(self, output_dir="./", prefix_filename=None, estimator_filter=None, parallelized=False, limit=None, export_configs=False):
    """
    Runs the given configurations, i.e.
    1) fits the estimator to the simulation and
    2) executes goodness-of-fit (currently: e.g. kl-divergence, wasserstein-distance etc.) tests
    Every successful run yields a result object of type GoodnessOfFitResult which contains information on both estimator, simulator and chosen hyperparameters
    such as n_samples, see GoodnessOfFitResult documentation for more information.

      Args:
        tasks: a list containing k tuples, each tuple has the shape (estimator object, simulator object)
        estimator_filter: a parameter to decide whether to execute just a specific type of estimator, e.g. "KernelMixtureNetwork",
                          must be one of the density estimator class types
        limit: limit the number of (potentially filtered) tasks
        parallelized: if True, the configurations are run in parallel mode on all available cpu's
        export_configs: determines if estimator configurations (weights etc.) should be exported to output dir, too

      Returns:
         a list of GoodnessOfFitResults objects (one per configuration run) and a list of the GoodnessOfFit objects (one per configuration run)
        which contain the fitted estimators
    """
    assert len(self.configs) > 0
    if estimator_filter is not None:
      self.configs = [tupl for tupl in self.configs if estimator_filter in tupl[0].__class__.__name__]
    assert len(self.configs), "no tasks to execute after filtering for the estimator"
    print("Running configurations. Number of total tasks after filtering: ", len(self.configs))

    if limit is not None:
      assert limit > 0, "limit mustn't be negative"
    else:
      limit = len(self.configs)

    config_file_name = "configurations"
    result_file_name = "result"
    if prefix_filename is not None:
      config_file_name = prefix_filename + "_" + config_file_name + "_"
      result_file_name = prefix_filename + "_" + result_file_name + "_"

    if parallelized:
      # todo: come up with a work-around for nested parallelized loops and tensorflow non-pickable objects
      with self._poolcontext(processes=None) as pool:
        gof_objects, gof_results = pool.starmap(self._run_single_configuration, self.configs[:limit])
        return gof_objects, gof_results

    else:
      if export_configs:
        file_configurations = io.get_full_path(output_dir=output_dir, suffix=".pickle", file_name=config_file_name)
        file_handle_configs = open(file_configurations, "a+b")

      file_results = io.get_full_path(output_dir=output_dir, suffix=".csv", file_name=result_file_name)
      file_handle_results = open(file_results, "a+")

      index = 0
      for i, task in enumerate(self.configs[:limit]):
        try:
          print("Task:", i+1, "Estimator:", task[0].__class__.__name__, " Simulator: ", task[1].__class__.__name__)
          gof_object, gof_result = self._run_single_configuration(*task)

          gof_result.x_cond = gof_result.x_cond.flatten()

          if export_configs:
            self._export_results(task=task, index=index, gof_result=gof_result, file_handle_results=file_handle_results,
                                 gof_object=gof_object, file_handle_configs=file_handle_configs)
          else:
            self._export_results(task=task, index=index, gof_result=gof_result, file_handle_results=file_handle_results)

          index = i + gof_result.n_x_cond

          """ write to file batch-wise to prevent memory overflow """
          if i % 50 == 0:
           file_handle_results.close()
           file_handle_results = open(file_results, "a+")

           if export_configs:
             file_handle_configs.close()
             file_handle_configs = open(file_configurations, "a+b")

          gc.collect()

        except Exception as e:
          print("error in task: ", i+1, " configuration: ", task)
          print(str(e))
          traceback.print_exc()


  def _run_single_configuration(self, estimator, simulator, n_observations, n_x_cond=5):
    gof = GoodnessOfFit(estimator=estimator, probabilistic_model=simulator, n_observations=n_observations, n_x_cond=n_x_cond)
    return gof, gof.compute_results()


  def _get_results_dataframe(self, results):
    """
    retrieves the dataframe for one or more GoodnessOfFitResults result objects.
      Args:
          results: a list or single object of type GoodnessOfFitResults
      Returns:
         a pandas dataframe
    """
    n_results = len(results)
    assert n_results > 0, "no results given"
    columns = ['estimator', 'simulator', 'n_observations', 'center_sampling_method', 'ndim_x', 'ndim_y', 'n_centers', 'kl_divergence', 'hellinger_distance',
               'js_divergence', 'x_cond']


    result_dicts = results.report_dict()

    return pd.DataFrame(result_dicts, columns=columns)

  def _export_results(self, task, index, gof_result, file_handle_results, gof_object=None, file_handle_configs=None):
    assert len(gof_result) > 0, "no results given"

    """ write result to file"""
    try:
      gof_result_df = self._get_results_dataframe(results=gof_result)
      io.append_result_to_csv(file_handle_results, gof_result_df, index)
    except Exception as e:
      print("appending to file was not successful for task: ", task)
      print(str(e))
      traceback.print_exc()

    if file_handle_configs and gof_object:
      """ write config to file"""
      try:
       io.append_obj_to_pickle(obj=gof_object, file_handle=file_handle_configs)
      except Exception as e:
       print("appending to file was not successful for task: ", task)
       print(str(e))
       traceback.print_exc()


  def _merge_names(self, a, b):
      return '{} & {}'.format(a, b)


  def _merge_names_unpack(self, args):
      return self._merge_names(*args)


  @contextmanager
  def _poolcontext(*args, **kwargs):
      pool = multiprocessing.Pool(*args, **kwargs)
      yield pool
      pool.terminate()

