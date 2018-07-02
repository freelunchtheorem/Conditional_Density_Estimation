import itertools
import pandas as pd
import numpy as np
import gc
import copy
import traceback
import logging

""" do not remove, imports required for globals() call """
from cde.density_estimator import LSConditionalDensityEstimation, KernelMixtureNetwork, MixtureDensityNetwork, ConditionalKernelDensityEstimation
from cde.density_simulation import EconDensity, GaussianMixture, ArmaJump, JumpDiffusionModel
from cde.evaluation.GoodnessOfFit import GoodnessOfFit, sample_x_cond
from cde.evaluation.GoodnessOfFitResults import GoodnessOfFitResults
from cde.utils import io
import hashlib
import base64
import pickle
import tensorflow as tf
import os
from ml_logger import logger
import config
import concurrent.futures

EXP_CONFIG_FILE = 'exp_configs.pkl'
RESULTS_FILE = 'results.pkl'

class ConfigRunner():
  """
  Args:
    exp_prefix: (str) prefix of experiment configuration
    est_params: dict containing estimator parametrization
                example:

                { 'KernelMixtureNetwork':
                          {'center_sampling_method': ["k_means"],
                           'n_centers': [20],
                           ...
                           }
                  'MixtureDensityNetwork':
                          {
                          ...
                           }
                }


    sim_params: diction containing simulator parametrization
                example:

                {'EconDensity': {'std': [1],
                                'heteroscedastic': [True]
                                },

                'GaussianMixture': { ... }
                }

    n_observations: either a array-like or a scalar value that defines the number of observations from the
                    simulation model that are used to train the estimators

    keys_of_interest: list of strings, each representing a column in the dataframe / csv export

    n_mc_samples: number of samples used for monte carlo sampling (a warning is printed if n_mc_samples is less than 10**5

    n_x_cond: (int) number of x conditionals to be sampled

    n_seeds: (int) number of different seeds for sampling the data
  """

  def __init__(self, exp_prefix, est_params, sim_params, observations, keys_of_interest, n_mc_samples=10 ** 7, n_x_cond=5, n_seeds=5, n_workers=None):
    assert est_params and exp_prefix and sim_params and keys_of_interest
    assert observations.all()

    sim_params = _add_seeds_to_sim_params(n_seeds, sim_params)

    self.observations = observations
    self.n_mc_samples = n_mc_samples
    self.n_x_cond = n_x_cond
    self.keys_of_interest = keys_of_interest
    self.exp_prefix = exp_prefix
    self.n_workers = n_workers

    logger.configure(config.DATA_DIR, prefix=exp_prefix, color='green')
    logger.log('INITIALIZING CONFIG RUNNER')

    ''' ---------- Either load or generate the configs ----------'''
    config_pkl_path = os.path.join(logger.log_directory, EXP_CONFIG_FILE)

    if os.path.isfile(config_pkl_path):
      logger.log("{:<70s} {:<30s}".format("Loading experiment previous configs from file: ", self.config_pickle_file))
      self.configs = logger.load_pkl(EXP_CONFIG_FILE)
    else:
      logger.log("{:<70s} {:<30s}".format("Generating and storing experiment configs under: ", config_pkl_path))
      self.configs = self._generate_configuration_variants(est_params, sim_params)
      logger.log_data(path=EXP_CONFIG_FILE, data=self.configs)

    ''' ---------- Either load already existing results or start a new result collection ---------- ''' #TODO: make this workflow with the logger
    results_pkl_path = os.path.join(logger.log_directory, logger.prefix, RESULTS_FILE)
    if os.path.isfile(results_pkl_path):
      logger.log("{:<70s} {:<30s}".format("Continue with: ", results_pkl_path))
      self.gof_single_res_collection = dict(logger.load_data(RESULTS_FILE))
    else: # start from scratch
      self.gof_single_res_collection = {}
    self.gof_results = GoodnessOfFitResults(self.gof_single_res_collection)


  def _generate_configuration_variants(self, est_params, sim_params):
    """
    Creates all possible combinations from the (configured) estimators and simulators.
    Requires configured estimators and simulators in the constructor:

    Args:
        est_params: estimator parameters as dict with 2 levels
        sim_params: density simulator parameters as dict with 2 levels

    Returns:
        if n_observations is not a list, a list containing n*m=k tuples while k being the number of the cartesian product of estimators and simulators is
        returned --> shape of tuples: (estimator object, simulator object)
        if n_observations is a list, n*m*o=k while o is the number of elements in n_observatons list
    """

    self.est_configs = _create_configurations(est_params)
    self.sim_configs = _create_configurations(sim_params)

    if np.isscalar(self.observations):
      self.observations = [self.observations]

    configs = []
    configured_sims = []

    """ since simulator configurations of the same kind require the same X,Y and x_cond, 
    they have to be generated separately from the estimators"""
    for simulator_name, sim_params in self.sim_configs.items():
      for config in sim_params:
        sim = globals()[simulator_name](**config)

        n_obs_max = max(self.observations)
        X_max, Y_max = sim.simulate(n_obs_max)
        X_max, Y_max = sim._handle_input_dimensionality(X_max, Y_max)

        for obs in self.observations:
          X, Y = X_max[:obs], Y_max[:obs]
          x_cond = sample_x_cond(X=X_max, n_x_cond=self.n_x_cond)
          configured_sims.append(dict({"simulator_name": simulator_name, 'simulator_config': config, "n_obs": obs, "X": X, "Y": Y, "x_cond": x_cond}))

    # merge simulator variants together with estimator variants
    task_number = 0
    for sim_dict in configured_sims:
      for estimator_name, estimator_params in self.est_configs.items() :
        for config in estimator_params:
          simulator_dict = copy.deepcopy(sim_dict)

          simulator_dict['estimator_name'] = estimator_name
          simulator_dict['estimator_config'] = config
          simulator_dict['task_name'] = '%s_task_%i'%(estimator_name, task_number)

          simulator_dict["n_mc_samples"] = self.n_mc_samples
          configs.append(simulator_dict)
          task_number += 1

    return configs

  def run_configurations(self, estimator_filter=None,
                         limit=None, dump_models=False):
    """
    Runs the given configurations, i.e.
    1) fits the estimator to the simulation and
    2) executes goodness-of-fit (currently: e.g. kl-divergence, wasserstein-distance etc.) tests
    Every successful run yields a result object of type GoodnessOfFitResult which contains
    information on both estimator, simulator and chosen hyperparameters

    such as n_samples, see GoodnessOfFitResult documentation for more information.

      Args:
        estimator_filter: a parameter to decide whether to execute just a specific type of estimator, e.g. "KernelMixtureNetwork",
                          must be one of the density estimator class types
        limit: limit the number of (potentially filtered) tasks
        dump_models: (boolean) whether to save/dump the fitted estimators

      Returns:
         returns two objects: (result_list, full_df)
          1) a GoodnessOfFitResults object containing all configurations as GoodnessOfFitSingleResult objects, carrying information about the
          estimator and simulator hyperparameters as well as n_obs, n_x_cond, n_mc_samples and the statistic results.
          2) a full pandas dataframe of the csv
          Additionally, if export_pickle is True, the path to the pickle file will be returned, i.e. return values are (results_list, full_df, path_to_pickle)

    """

    self.dump_models = dump_models
    ''' Asserts, Setup and Applying Filters/Limits  '''
    assert len(self.configs) > 0
    self._apply_filters(estimator_filter)

    if limit is not None:
      assert limit > 0, "limit must not be negative"
      logger.log("Limit enabled. Running only the first {} configurations".format(limit))

    ''' Run the configurations '''

    logger.log("{:<70s} {:<30s}".format("Number of total tasks in pipeline:", str(len(self.configs))))
    logger.log("{:<70s} {:<30s}".format("Number of aleady finished tasks (found in results pickle): ", str(len(self.gof_single_res_collection))))

    tasks = self.configs[:limit]
    iters = range(1, len(tasks)+1)

    with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_workers) as executor:
      for i, task_result in zip(iters, executor.map(self._run_single_task, iters, tasks)):
        if task_result and type(task_result) is tuple:
          task_hash, gof_single_result = task_result
          self.gof_single_res_collection[task_hash] = gof_single_result

    #TODO add csv logging
    #self._dump_current_state(task, gof_single_result)

    gof_results = GoodnessOfFitResults(single_results_dict=self.gof_single_res_collection)
    full_df = gof_results.generate_results_dataframe(keys_of_interest=self.keys_of_interest)

    return self.gof_single_res_collection, full_df

  def _run_single_task(self, i, task):
    try:
      task_hash = _hash_task_dict(task)  # generate SHA256 hash of task dict as identifier

      if task_hash in self.gof_single_res_collection.keys():
        logger.log("Task {:<1} {:<63} {:<10} {:<1} {:<1} {:<1}".format(i + 1, "has already been completed:", "Estimator:",
                                                                       task['estimator_name'],
                                                                       " Simulator: ", task["simulator_name"]))
        return None

      else:
        logger.log(
          "Task {:<1} {:<63} {:<10} {:<1} {:<1} {:<1}".format(i + 1, "running:", "Estimator:", task['estimator_name'],
                                                              " Simulator: ", task["simulator_name"]))

        tf.reset_default_graph()

        ''' build simulator and estimator model given the specified configurations '''

        simulator = globals()[task['simulator_name']](**task['simulator_config'])

        estimator = globals()[task['estimator_name']](task['task_name'], simulator.ndim_x,
                                                      simulator.ndim_y, **task['estimator_config'])

        with tf.Session() as sess:
          sess.run(tf.global_variables_initializer())

          ''' train the model '''
          gof = GoodnessOfFit(estimator=estimator, probabilistic_model=simulator, X=task['X'], Y=task['Y'],
                              n_observations=task['n_obs'],
                              n_mc_samples=task['n_mc_samples'], x_cond=task['x_cond'])

          gof.fit_estimator(print_fit_result=False)

          if self.dump_models:
            logger.dump_data(path="model_dumps/{}.pkl".format(task['task_name']), data=gof.estimator)

          ''' perform tests with the fitted model '''
          gof_results = gof.compute_results()
          gof_results.task_name = task['task_name']

          gof_results.hash = task_hash

        logger.log_data(RESULTS_FILE, (task_hash, gof_results))

        return gof_results, task_hash

    except Exception as e:
      logger.log("error in task: ", i + 1)
      logger.log(str(e))
      traceback.print_exc()

  def _dump_current_state(self, task, gof_single_result):
    if self.export_csv:
      self._export_results(task=task, gof_result=gof_single_result, file_handle_results=self.file_handle_results_csv)
    if self.export_pickle:
      with open(self.results_pickle_path, "wb") as f:
        intermediate_gof_results = GoodnessOfFitResults(single_results_dict=self.gof_single_res_collection)
        io.dump_as_pickle(f, intermediate_gof_results, verbose=False)


  def _get_results_dataframe(self, results):
    """ retrieves the dataframe for one or more GoodnessOfFitResults result objects.

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

  def _apply_filters(self, estimator_filter):

    if estimator_filter is not None:
      self.configs = [tupl for tupl in self.configs if estimator_filter in tupl["estimator"].__class__.__name__]

    if len(self.configs) == 0:
      print("no tasks to execute after filtering for the estimator")
      return None

  def _setup_file_names(self):
    if self.prefix_filename is not None:
      self.result_file_name = self.prefix_filename + "_" + self.result_file_name + "_"

    if self.export_pickle:
      if self.results_pickle_file: # continue with old file
        self.results_pickle_path = self.results_pickle_file
      else: # new file name
        self.results_pickle_path = io.get_full_path(output_dir=self.output_dir, suffix=".pickle", file_name=self.result_file_name)

    if self.export_csv:
      if self.results_pickle_file:
        self.results_csv_path = self.results_pickle_file.replace("pickle", "csv")
      else:
        self.results_csv_path = io.get_full_path(output_dir=self.output_dir, suffix=".csv", file_name=self.result_file_name)
      self.file_handle_results_csv = open(self.results_csv_path, "a+")

    if self.dump_models:
      self.model_dump_dir = os.path.join(self.output_dir, 'model_dumps')
      if not os.path.exists(self.model_dump_dir):
        os.makedirs(self.model_dump_dir)



def _add_seeds_to_sim_params(n_seeds, sim_params):
  seeds = [20 + i for i in range(n_seeds)]
  for sim_instance in sim_params.keys():
    sim_params[sim_instance]['random_seed'] = seeds
  return sim_params


def _create_configurations(params_dict):
  confs = {}
  for conf_instance, conf_dict in params_dict.items():
    conf_product = list(itertools.product(*list(conf_dict.values())))
    conf_product_dicts = [(dict(zip(conf_dict.keys(), conf))) for conf in conf_product]
    confs[conf_instance] = conf_product_dicts

  return confs

# TODO: replace SHA256 by faster, non-cryptographic hash function (e.g. pyhashxx)
def _hash_task_dict(task_dict):
  assert {'simulator_name', 'simulator_config', 'estimator_name', 'estimator_config', 'x_cond', 'n_mc_samples', 'n_obs'} < set(task_dict.keys())
  task_dict = copy.deepcopy(task_dict)

  tpls = _make_hashable(task_dict)
  return make_hash_sha256(tpls)

def _make_hashable(o):
    if isinstance(o, (tuple, list)):
        return tuple((_make_hashable(e) for e in o))
    if isinstance(o, dict):
        return tuple(sorted((k, _make_hashable(v)) for k, v in o.items()))

    if isinstance(o, (set, frozenset)):
        return tuple(sorted(_make_hashable(e) for e in o))
    return o

def make_hash_sha256(o):
    hasher = hashlib.sha256()
    hasher.update(repr(_make_hashable(o)).encode())
    return base64.b64encode(hasher.digest()).decode()

