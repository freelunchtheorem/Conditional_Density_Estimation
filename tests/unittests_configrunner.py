import random
import unittest
import config
import shutil
import tensorflow as tf
import sys
import os
from ml_logger import logger

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from cde.evaluation.ConfigRunner import ConfigRunner
from cde.evaluation_runs.question1_noise_reg_xy import question1

NUM_CONFIGS_TO_TEST = 1

EXP_PREFIX = 'test_io_question1_noise_reg_x'
EXP_CONFIG_FILE = 'exp_configs.pkl'
RESULTS_FILE = 'results.pkl'

class configrunner(unittest.TestCase):

  def test_store_load_configrunner_pipeline(self):

    logger.configure(log_directory=config.DATA_DIR, prefix=EXP_PREFIX)
    test_dir = os.path.join(logger.log_directory, logger.prefix)
    if os.path.exists(test_dir):
      shutil.rmtree(test_dir)


    keys_of_interest = ['task_name', 'estimator', 'simulator', 'n_observations', 'center_sampling_method', 'x_noise_std', 'y_noise_std',
                        'ndim_x', 'ndim_y', 'n_centers', "n_mc_samples", "n_x_cond", 'mean_est', 'cov_est', 'mean_sim', 'cov_sim',
                        'kl_divergence', 'hellinger_distance', 'js_divergence', 'x_cond', 'random_seed', "mean_sim", "cov_sim",
                        "mean_abs_diff", "cov_abs_diff", "VaR_sim", "VaR_est", "VaR_abs_diff", "CVaR_sim", "CVaR_est", "CVaR_abs_diff",
                        "time_to_fit"]


    conf_est, conf_sim, observations = question1()
    conf_runner = ConfigRunner(EXP_PREFIX, conf_est, conf_sim, observations=observations, keys_of_interest=keys_of_interest,
                               n_mc_samples=1 * 10 ** 2, n_x_cond=5, n_seeds=5)

    conf_runner.configs = random.sample(conf_runner.configs, NUM_CONFIGS_TO_TEST)

    results_from_configrunner, _ = conf_runner.run_configurations(dump_models=True, multiprocessing=False)
    results_from_pkl_file = dict(logger.load_pkl(RESULTS_FILE))

    """ check results pickle """
    self.assertTrue(results_from_pkl_file.keys() == results_from_configrunner.keys())

    """ check if model dumps have all been created """
    dump_dir = os.path.join(logger.log_directory, logger.prefix, 'model_dumps')
    model_dumps_list = os.listdir(dump_dir) # get list of all model files
    model_dumps_list_no_suffix = [os.path.splitext(entry)[0] for entry in model_dumps_list] # remove suffix

    for conf in conf_runner.configs:
      self.assertTrue(conf['task_name'] in model_dumps_list_no_suffix)


    """ check if model dumps can be used successfully"""
    for model_dump_i in model_dumps_list:
      #tf.reset_default_graph()
      with tf.Session(graph=tf.Graph()):
        model = logger.load_pkl("model_dumps/"+model_dump_i)
        self.assertTrue(model)
        if model.ndim_x == 1 and model.ndim_y == 1:
          self.assertTrue(model.plot3d(show=False))



if __name__ == '__main__':

  testmodules = ['unittests_configrunner.configrunner']
  suite = unittest.TestSuite()
  for t in testmodules:
    try:
      # If the module defines a suite() function, call it to get the suite.
      mod = __import__(t, globals(), locals(), ['suite'])
      suitefn = getattr(mod, 'suite')
      suite.addTest(suitefn())
    except (ImportError, AttributeError):
      # else, just load all the test cases from the module.
      suite.addTest(unittest.defaultTestLoader.loadTestsFromName(t))

  unittest.TextTestRunner().run(suite)