import itertools
import pandas as pd
import numpy as np
import gc
import copy
import traceback
import logging

""" do not remove, imports required for globals() call """
from cde.density_estimator import LSConditionalDensityEstimation, KernelMixtureNetwork, MixtureDensityNetwork
from cde.density_simulation import EconDensity, GaussianMixture
from cde.evaluation.GoodnessOfFit import GoodnessOfFit, sample_x_cond
from cde.evaluation.GoodnessOfFitResults import GoodnessOfFitResults
from cde.utils import io





class ConfigRunner():
  """
  Args:
    est_params: a list of estimator objects with length n (the number of configured estimators)
    sim_params: a list of instantiated simulator objects with length n (the number of configured simulators)
    n_observations: n_observations: either a list or a scalar value that defines the number of observations from the
    simulation model that are used to train the estimators
    keys_of_interest: list of strings, each representing a column in the dataframe / csv export
    n_mc_samples: number of samples used for monte carlo sampling (a warning is printed if n_mc_samples is less than 10**5
    n_x_cond: number of x conditionals to be sampled
  """

  def __init__(self, est_params, sim_params, observations, keys_of_interest, n_mc_samples=10 ** 6, n_x_cond=5):
    assert est_params
    assert sim_params
    assert keys_of_interest
    assert observations.all()

    logging.log(logging.INFO, "creating configurations...")
    self.observations = observations
    self.configured_estimators = [globals()[estimator_name](*config) for estimator_name, estimator_params in est_params.items() for config in estimator_params]
    self.configured_simulators = [globals()[simulator_name](*sim_params) for simulator_name, sim_params in sim_params.items()]

    self.n_mc_samples = n_mc_samples
    self.n_x_cond = n_x_cond

    self.configs = self._create_configurations()
    self.keys_of_interest = keys_of_interest



  def _create_configurations(self):
    """
    Creates all possible combinations from the (configured) estimators and simulators.
    Requires configured estimators and simulators in the constructor:

    Returns:
        if n_observations is not a list, a list containing n*m=k tuples while k being the number of the cartesian product of estimators and simulators is
        returned --> shape of tuples: (estimator object, simulator object)
        if n_observations is a list, n*m*o=k while o is the number of elements in n_observatons list
    """
    print("total number of configurations to be generated: " +
          str(len(self.configured_estimators) * len(self.configured_simulators) * len(self.observations)))


    if np.isscalar(self.observations):
      self.observations = [self.observations]

    configs = []
    configured_sims = []

    """ since simulator configurations of the same kind require the same X,Y and x_cond, 
    they have to be generated separately from the estimators"""
    for sim in self.configured_simulators:
      n_obs_max = max(self.observations)
      X_max, Y_max = sim.simulate(n_obs_max)
      X_max, Y_max = sim._handle_input_dimensionality(X_max, Y_max)

      for obs in self.observations:
        X, Y = X_max[:obs], Y_max[:obs]
        x_cond = sample_x_cond(X=X, n_x_cond=self.n_x_cond)
        configured_sims.append(dict({"simulator": sim, "n_obs": obs, "X": X, "Y": Y, "x_cond": x_cond}))


    for estimator, simulator in itertools.product(self.configured_estimators, configured_sims):
      simulator["estimator"] = estimator
      simulator["n_mc_samples"] = self.n_mc_samples
      configs.append(copy.deepcopy(simulator))


    return configs




  def run_configurations(self, export_csv = True, output_dir="./", prefix_filename=None, estimator_filter=None, limit=None, export_pickle=True):
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
        export_pickle: determines if all results should be exported to output dir as pickle in addition to the csv

      Returns:
         returns two objects: (result_list, full_df)
          1) a GoodnessOfFitResults object containing all configurations as GoodnessOfFitSingleResult objects, carrying information about the
          estimator and simulator hyperparameters as well as n_obs, n_x_cond, n_mc_samples and the statistic results.
          2) a full pandas dataframe of the csv
          Additionally, if export_pickle is True, the path to the pickle file will be returned, i.e. return values are (results_list, full_df, path_to_pickle)

    """
    assert len(self.configs) > 0
    if estimator_filter is not None:
      self.configs = [tupl for tupl in self.configs if estimator_filter in tupl["estimator"].__class__.__name__]
    if len(self.configs) == 0:
      print("no tasks to execute after filtering for the estimator")
      return
    print("Running configurations. Number of total tasks after filtering: ", len(self.configs))

    if limit is not None:
      assert limit > 0, "limit must not be negative"
      print("Limit enabled. Running only the first {} configurations".format(limit))

    config_file_name = "configurations"
    result_file_name = "result"
    if prefix_filename is not None:
      config_file_name = prefix_filename + "_" + config_file_name + "_"
      result_file_name = prefix_filename + "_" + result_file_name + "_"


    if export_pickle:
      results_pickle = io.get_full_path(output_dir=output_dir, suffix=".pickle", file_name=config_file_name)
      file_handle_results_pickle = open(results_pickle, "a+b")

    if export_csv:
      file_results = io.get_full_path(output_dir=output_dir, suffix=".csv", file_name=result_file_name)
      file_handle_results_csv = open(file_results, "a+")

    gof_single_res_collection = []

    for i, task in enumerate(self.configs[:limit]):
      try:
        print("Task:", i+1, "Estimator:", task["estimator"].__class__.__name__, " Simulator: ", task["simulator"].__class__.__name__)
        _, gof_single_result = self._run_single_configuration(**task)

        if export_csv:
          self._export_results(task=task, gof_result=gof_single_result, file_handle_results=file_handle_results_csv)

        gof_single_res_collection.append(gof_single_result)

      except Exception as e:
        print("error in task: ", i+1, " configuration: ", task)
        print(str(e))
        traceback.print_exc()

    gof_results = GoodnessOfFitResults(single_results_list=gof_single_res_collection)
    full_df = gof_results.generate_results_dataframe(keys_of_interest=self.keys_of_interest)

    if export_pickle:
      io.dump_as_pickle(file_handle_results_pickle, gof_results)
      pickle_path = file_handle_results_pickle.name
      file_handle_results_pickle.close()
      return gof_single_res_collection, full_df, pickle_path

    file_handle_results_csv.close()
    return gof_single_res_collection, full_df

  def _run_single_configuration(self, estimator, simulator, X, Y, x_cond, n_obs, n_mc_samples):
    gof = GoodnessOfFit(estimator=estimator, probabilistic_model=simulator, X=X, Y=Y, n_observations=n_obs,
                        n_mc_samples=n_mc_samples, x_cond=x_cond)
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

    results_dict = results.report_dict(keys_of_interest=self.keys_of_interest)


    return pd.DataFrame.from_dict(data=results_dict)


  def _export_results(self, task, gof_result, file_handle_results):
    assert len(gof_result) > 0, "no results given"

    """ write result to file"""
    try:
      gof_result_df = self._get_results_dataframe(results=gof_result)
      gof_result.result_df = gof_result_df
      io.append_result_to_csv(file_handle_results, gof_result_df)
    except Exception as e:
      print("appending to file was not successful for task: ", task)
      print(str(e))
      traceback.print_exc()



