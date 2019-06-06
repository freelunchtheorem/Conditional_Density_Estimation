import itertools
import pandas as pd
import numpy as np
import copy
import traceback

""" do not remove, imports required for globals() call """
from cde.density_estimator import LSConditionalDensityEstimation, KernelMixtureNetwork, MixtureDensityNetwork, ConditionalKernelDensityEstimation, NeighborKernelDensityEstimation, NormalizingFlowEstimator
from cde.density_simulation import EconDensity, GaussianMixture, ArmaJump, JumpDiffusionModel, SkewNormal, LinearGaussian, LinearStudentT
from cde.model_fitting.GoodnessOfFitLogProb import GoodnessOfFitLogProb
from cde.model_fitting.GoodnessOfFitResults import GoodnessOfFitResults
from cde.model_fitting.ConfigRunner import load_dumped_estimators, _hash_task_dict, _create_configurations, _add_seeds_to_sim_params
from cde.utils import io
from cde.utils.async_executor import AsyncExecutor
from ml_logger import logger
import tensorflow as tf
import os
import config
import time

EXP_CONFIG_FILE = 'exp_configs.pkl'
RESULTS_FILE = 'results.pkl'

class ConfigRunnerLogProb():
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


    sim_params: dict containing simulator parametrization
                example:

                {'EconDensity': {'std': [1],
                                'heteroscedastic': [True]
                                },

                'GaussianMixture': { ... }
                }

    observations: either a array-like or a scalar value that defines the number of observations from the
                    simulation model that are used to train the estimators

    keys_of_interest: list of strings, each representing a column in the dataframe / csv export

    n_test_samples: number of samples used to compute test score

    n_seeds: (int) number of different seeds for sampling the data
  """

  def __init__(self, exp_prefix, est_params, sim_params, observations, keys_of_interest, n_test_samples=10 ** 5,
               n_seeds=5, use_gpu=True):

    assert est_params and exp_prefix and sim_params and keys_of_interest
    assert observations.all()

    # convert to dicts to list of tuples
    if isinstance(est_params, dict):
      est_params = list(est_params.items())

    if isinstance(sim_params, dict):
      sim_params = list(sim_params.items())

    # every simulator configuration will be run multiple times with different randomness seeds
    sim_params = _add_seeds_to_sim_params(n_seeds, sim_params)

    self.observations = observations
    self.n_test_samples = n_test_samples
    self.keys_of_interest = keys_of_interest
    self.exp_prefix = exp_prefix
    self.use_gpu = use_gpu

    logger.configure(log_directory=config.DATA_DIR, prefix=exp_prefix, color='green')

    ''' ---------- Either load or generate the configs ----------'''
    config_pkl_path = os.path.join(logger.log_directory, logger.prefix, EXP_CONFIG_FILE)

    if os.path.isfile(config_pkl_path):
      logger.log("{:<70s} {:<30s}".format("Loading experiment previous configs from file: ", config_pkl_path))
      self.configs = logger.load_pkl(EXP_CONFIG_FILE)
    else:
      logger.log("{:<70s} {:<30s}".format("Generating and storing experiment configs under: ", config_pkl_path))
      self.configs = self._generate_configuration_variants(est_params, sim_params)
      logger.dump_pkl(data=self.configs, path=EXP_CONFIG_FILE)

    ''' ---------- Either load already existing results or start a new result collection ---------- '''
    results_pkl_path = os.path.join(logger.log_directory, logger.prefix, RESULTS_FILE)
    if os.path.isfile(results_pkl_path):
      logger.log_line("{:<70s} {:<30s}".format("Continue with: ", results_pkl_path))
      self.gof_single_res_collection = dict(logger.load_pkl_log(RESULTS_FILE))

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
        X_test, Y_test = sim.simulate(self.n_test_samples)

        for obs in self.observations:
          X, Y = X_max[:obs], Y_max[:obs]
          configured_sims.append(dict({"simulator_name": simulator_name, 'simulator_config': config, "n_obs": obs,
                                       "X": X, "Y": Y, "X_test": X_test, "Y_test": Y_test}))

    # merge simulator variants together with estimator variants
    task_number = 0
    for sim_dict in configured_sims:
      for estimator_name, estimator_params in self.est_configs.items() :
        for config in estimator_params:
          simulator_dict = copy.deepcopy(sim_dict)

          simulator_dict['estimator_name'] = estimator_name
          simulator_dict['estimator_config'] = config
          simulator_dict['task_name'] = '%s_task_%i'%(estimator_name, task_number)
          configs.append(simulator_dict)
          task_number += 1

    return configs

  def run_configurations(self, dump_models=False, multiprocessing=True, n_workers=None):
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

    ''' Asserts '''
    assert len(self.configs) > 0
    tasks = self.configs

    ''' Run the configurations '''

    logger.log("{:<70s} {:<30s}".format("Number of total tasks in pipeline:", str(len(self.configs))))
    logger.log("{:<70s} {:<30s}".format("Number of aleady finished tasks (found in results pickle): ",
                                         str(len(self.gof_single_res_collection))))


    iters = range(len(tasks))

    if multiprocessing:
      executor = AsyncExecutor(n_jobs=n_workers)
      executor.run(self._run_single_task, iters, tasks)

    else:
      for i, task in zip(iters, tasks):
        self._run_single_task(i, task)


  def _run_single_task(self, i, task):
    start_time = time.time()
    try:
      task_hash = _hash_task_dict(task)  # generate SHA256 hash of task dict as identifier

      # skip task if it has already been completed
      if task_hash in self.gof_single_res_collection.keys():
        logger.log("Task {:<1} {:<63} {:<10} {:<1} {:<1} {:<1}".format(i + 1, "has already been completed:", "Estimator:",
                                                                       task['estimator_name'],
                                                                       " Simulator: ", task["simulator_name"]))
        return None

      # run task when it has not been completed
      else:
        logger.log(
          "Task {:<1} {:<63} {:<10} {:<1} {:<1} {:<1}".format(i + 1, "running:", "Estimator:", task['estimator_name'],
                                                              " Simulator: ", task["simulator_name"]))

        tf.reset_default_graph()

        ''' build simulator and estimator model given the specified configurations '''

        simulator = globals()[task['simulator_name']](**task['simulator_config'])

        t = time.time()
        estimator = globals()[task['estimator_name']](task['task_name'], simulator.ndim_x,
                                                      simulator.ndim_y, **task['estimator_config'])
        time_to_initialize = time.time() - t

        # if desired hide gpu devices
        if not self.use_gpu:
          os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        with tf.Session() as sess:
          sess.run(tf.global_variables_initializer())

          ''' train the model '''
          gof = GoodnessOfFitLogProb(estimator=estimator, probabilistic_model=simulator, X_train=task['X'], Y_train=task['Y'],
                                     X_test=task['X_test'], Y_test=task['Y_test'], task_name = task['task_name'])

          t = time.time()
          gof.fit_estimator(print_fit_result=True)
          time_to_fit = time.time() - t

          if self.dump_models:
            logger.dump_pkl(data=gof.estimator, path="model_dumps/{}.pkl".format(task['task_name']))
            logger.dump_pkl(data=gof.probabilistic_model, path="model_dumps/{}.pkl".format(task['task_name'] + "_simulator"))

          ''' perform tests with the fitted model '''
          t = time.time()
          gof_results = gof.compute_results()
          time_to_evaluate = time.time() - t

          gof_results.task_name = task['task_name']

          gof_results.hash = task_hash

        logger.log_pkl(data=(task_hash, gof_results), path=RESULTS_FILE)
        logger.flush(file_name=RESULTS_FILE)
        del gof_results

        task_duration = time.time() - start_time
        logger.log(
          "Finished task {:<1} in {:<1.4f} {:<43} {:<10} {:<1} {:<1} {:<2} | {:<1} {:<1.2f} {:<1} {:<1.2f} {:<1} {:<1.2f}".format(i + 1, task_duration, "sec:",
          "Estimator:", task['estimator_name'], " Simulator: ", task["simulator_name"], "t_init:", time_to_initialize, "t_fit:", time_to_fit, "t_eval:", time_to_evaluate))

    except Exception as e:
      logger.log("error in task: ", str(i + 1))
      logger.log(str(e))
      traceback.print_exc()

  def _dump_current_state(self):
    #if self.export_csv:
    #  self._export_results(task=task, gof_result=gof_single_result, file_handle_results=self.file_handle_results_csv)
    #if self.export_pickle:
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




