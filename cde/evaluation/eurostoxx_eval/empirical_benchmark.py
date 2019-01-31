from cde.density_estimator import KernelMixtureNetwork, ConditionalKernelDensityEstimation, MixtureDensityNetwork, \
  NeighborKernelDensityEstimation, LSConditionalDensityEstimation
from cde.evaluation.eurostoxx_eval.load_dataset import make_overall_eurostoxx_df, target_feature_split

from sklearn.model_selection import cross_validate
from cde.density_estimator import LSConditionalDensityEstimation, KernelMixtureNetwork, MixtureDensityNetwork, ConditionalKernelDensityEstimation, NeighborKernelDensityEstimation

from cde.model_fitting.ConfigRunner import _create_configurations
import numpy as np
import time
import pandas as pd
import argparse
import itertools
from collections import OrderedDict
from multiprocessing import Manager
from cde.utils.async_executor import AsyncExecutor

VALIDATION_PORTION = 0.2

ndim_x = 14
ndim_y = 1

N_SAMPLES = 10**5
SEEDS = [22, 23, 24, 25, 26]

VERBOSE = True

def train_valid_split(valid_portion):
  assert 0 < valid_portion < 1
  df = make_overall_eurostoxx_df()
  df = df.dropna()

  # split data into train and validation set
  split_index = int(df.shape[0] * (1.0 - valid_portion))

  df_train = df[:split_index]
  df_valid = df[split_index:]

  return df_train, df_valid


def empirical_evaluation(estimator, valid_portion=0.2, moment_r2=True, eval_by_fc=False, fit_by_cv=False):
  """
  Fits the estimator and, based on a left out validation splot, computes the
  Root Mean Squared Error (RMSE) between realized and estimated mean and std

  Args:
    estimator: estimator object
    valid_portion: portion of dataset to be separated as validation set
    moment_r2: (bool) whether to compute the rmse of mean and variance

  Returns:
    (likelihood, mu_rmse, std_rmse)
  """

  # get data and split into train and valid set
  df_train, df_valid = train_valid_split(valid_portion)

  X_train, Y_train = target_feature_split(df_train, 'log_ret_1', filter_nan=True)
  X_valid, Y_valid = target_feature_split(df_valid, 'log_ret_1', filter_nan=True)


  # realized moments
  mu_realized = df_valid['log_ret_last_period'][1:]
  std_realized_intraday = np.sqrt(df_valid['RealizedVariation'][1:])

  # fit density model
  if eval_by_fc and not fit_by_cv:
    raise NotImplementedError
  elif not eval_by_fc and fit_by_cv:
    estimator.fit_by_cv(X_train, Y_train, n_folds=5)
  else:
    estimator.fit(X_train, Y_train)


  # compute avg. log likelihood
  mean_logli = np.mean(estimator.log_pdf(X_valid, Y_valid))

  if moment_r2:
    # predict mean and std
    mu_predicted, std_predicted = estimator.mean_std(X_valid, n_samples=N_SAMPLES)
    mu_predicted = mu_predicted.flatten()[:-1]
    std_predicted = std_predicted.flatten()[:-1]

    assert mu_realized.shape == mu_predicted.shape
    assert std_realized_intraday.shape == std_realized_intraday.shape

    # compute realized std
    std_realized = np.abs(mu_predicted - mu_realized)

    # compute RMSE
    mu_rmse = np.sqrt(np.mean((mu_realized - mu_predicted) ** 2))
    std_rmse = np.sqrt(np.mean((std_realized - std_predicted) ** 2))
    std_intraday_rmse = np.sqrt(np.mean((std_realized_intraday - std_predicted) ** 2))
  else:
    mu_rmse, std_rmse, std_intraday_rmse = None, None, None

  return mean_logli, mu_rmse, std_rmse, std_intraday_rmse


def empirical_benchmark(model_dict, moment_r2=True, eval_by_fc=True, fit_by_cv=False, n_jobs=-1):
  result_dict = {}

  # multiprocessing setup
  manager = Manager()
  result_list_model = manager.list()
  if n_jobs == -1:
    n_jobs = len(SEEDS)
  executor = AsyncExecutor(n_jobs=n_jobs)
  eval = lambda est: result_list_model.append(empirical_evaluation(est, VALIDATION_PORTION, moment_r2=moment_r2,
                                                             eval_by_fc=eval_by_fc, fit_by_cv=fit_by_cv))

  for model_name, models in model_dict.items():
    print("Running likelihood fit and validation for %s" % model_name)
    t = time.time()

    # Multiprocessing calls
    executor.run(eval, models)

    assert len(result_list_model) == len(models)
    mean_logli_list, mu_rmse_list, std_rmse_list, std_intraday_rmse_list = list(zip(*list(result_list_model)))

    # clear result list
    for _ in range(len(result_list_model)):
      del result_list_model[0]
    assert len(result_list_model) == 0

    mean_logli, mean_logli_dev = np.mean(mean_logli_list), np.std(mean_logli_list)
    mu_rmse, mu_rmse_dev = np.mean(mu_rmse_list), np.std(mu_rmse_list)
    std_rmse, std_rmse_dev = np.mean(std_rmse_list), np.std(std_rmse_list)
    std_intraday_rmse, std_intraday_rmse_dev = np.mean(std_intraday_rmse_list), np.std(std_intraday_rmse_list)

    result_dict[model_name] = mean_logli, mean_logli_dev, mu_rmse, mu_rmse_dev, std_rmse, std_rmse_dev, std_intraday_rmse, std_intraday_rmse_dev
    print('%s results:' % model_name, result_dict[model_name])
    print('Duration of %s:' % model_name, time.time() - t)

  df = pd.DataFrame.from_dict(result_dict, 'index')
  df.columns = ['log_likelihood', 'log_likelihood_dev', 'rmse_mean', 'rmse_mean_dev', 'rmse_std', 'rmse_std_dev', 'rmse_std_intraday', 'rmse_std_intraday_dev']
  return df


def initialize_models(model_dict, verbose=False):
  ''' make kartesian product of listed parameters per model '''
  model_configs = {}
  for model_key, conf_dict in model_dict.items():
    print(model_key)
    model_configs[model_key] = [dict(zip(conf_dict.keys(), value_tuple)) for value_tuple in
                                list(itertools.product(*list(conf_dict.values())))]

  """ initialize models """
  configs_initialized = {}
  for model_key, model_conf_list in model_configs.items():
    configs_initialized[model_key] = []
    for i, conf in enumerate(model_conf_list):
      conf['name'] = model_key.replace(' ', '_') + '_%i' % i
      if verbose: print("instantiating ", conf['name'])
      """ remove estimator entry from dict to instantiate it"""
      estimator = conf.pop('estimator')
      configs_initialized[model_key].append(globals()[estimator](**conf))
  return configs_initialized


# run methods

def run_benchmark_train_test(n_jobs=-1):
  print("Normal fit & Evaluation")

  model_dict = {
    'CKDE': {'estimator': ['ConditionalKernelDensityEstimation'], 'ndim_x': [ndim_x], 'ndim_y': [ndim_y],
                               'bandwidth': ['normal_reference'], 'random_seed': [None]},

    'LSCDE': {'estimator': ['LSConditionalDensityEstimation'], 'ndim_x': [ndim_x], 'ndim_y': [ndim_y],
              'random_seed': SEEDS},

    'NKDE': {'estimator': ['NeighborKernelDensityEstimation'], 'ndim_x': [ndim_x], 'ndim_y': [ndim_y],
                  'param_selection': ['normal_reference'], 'random_seed': [None]},

    'MDN w/ noise': {'estimator': ['MixtureDensityNetwork'], 'ndim_x': [ndim_x], 'ndim_y': [ndim_y],
                     'n_centers': [10], 'n_training_epochs': [1000], 'x_noise_std': [0.2], 'y_noise_std': [0.1],
                     'random_seed': SEEDS},

    'MDN w/0 noise': {'estimator': ['MixtureDensityNetwork'], 'ndim_x': [ndim_x], 'ndim_y': [ndim_y],
                      'n_centers': [10], 'n_training_epochs': [1000], 'x_noise_std': [None], 'y_noise_std': [None],
                      'random_seed': SEEDS},

    'KMN w/ noise': {'estimator': ['KernelMixtureNetwork'], 'ndim_x': [ndim_x], 'ndim_y': [ndim_y], 'n_centers': [50],
                     'n_training_epochs': [1000], 'init_scales': [[0.7, 0.3]], 'x_noise_std': [0.2],
                     'y_noise_std': [0.1],
                     'random_seed': SEEDS},

    'KMN w/0 noise': {'estimator': ['KernelMixtureNetwork'], 'ndim_x': [ndim_x], 'ndim_y': [ndim_y], 'n_centers': [50],
                      'n_training_epochs': [1000], 'init_scales': [[0.7, 0.3]], 'x_noise_std': [None],
                      'y_noise_std': [None],
                      'random_seed': SEEDS},
  }

  model_dict = initialize_models(model_dict, verbose=VERBOSE)
  model_dict = OrderedDict(list(model_dict.items()))

  result_df = empirical_benchmark(model_dict, moment_r2=True, eval_by_fc=False, fit_by_cv=False, n_jobs=n_jobs)
  print(result_df.to_latex())
  print(result_df)

def run_benchmark_train_test_fit_by_cv(model_key=None, n_jobs=-1):
  print("Fit by cv & Evaluation")
  model_dict_fit_by_cv = {
    'LSCDE_cv': {'estimator': ['LSConditionalDensityEstimation'], 'ndim_x': [ndim_x], 'ndim_y': [ndim_y],
                 'random_seed': SEEDS},

    'MDN_cv': {'estimator': ['MixtureDensityNetwork'], 'ndim_x': [ndim_x], 'ndim_y': [ndim_y],
               'n_training_epochs': [1000], 'random_seed': SEEDS},

    'KMN_cv': {'estimator': ['KernelMixtureNetwork'], 'ndim_x': [ndim_x], 'ndim_y': [ndim_y],
               'n_training_epochs': [1000], 'init_scales': [[0.7, 0.3]], 'random_seed': SEEDS},
  }

  # exclude all other models except model_key
  if model_key in list(model_dict_fit_by_cv.keys()):
    model_dict_fit_by_cv = {model_key: model_dict_fit_by_cv[model_key]}

  model_dict = initialize_models(model_dict_fit_by_cv, verbose=VERBOSE)
  model_dict = OrderedDict(list(model_dict.items()))
  result_df = empirical_benchmark(model_dict, moment_r2=True, eval_by_fc=False, fit_by_cv=True, n_jobs=n_jobs)

  print(result_df.to_latex())
  print(result_df)

def run_benchmark_train_test_cv_ml(n_jobs=-1):
  print("Fit by cv_ml & Evaluation")
  model_dict_cv_ml = {
    'CKDE_cv_ml': {'estimator': ['ConditionalKernelDensityEstimation'], 'ndim_x': [ndim_x], 'ndim_y': [ndim_y],
                   'bandwidth': ['cv_ml'], 'random_seed': [22]},

    'NKDE_cv_ml': {'estimator': ['NeighborKernelDensityEstimation'], 'ndim_x': [ndim_x], 'ndim_y': [ndim_y],
                  'param_selection': ['cv_ml'], 'random_seed': [22]},
  }

  model_dict = initialize_models(model_dict_cv_ml, verbose=VERBOSE)
  model_dict = OrderedDict(list(model_dict.items()))
  result_df_cv_ml = empirical_benchmark(model_dict, moment_r2=True, eval_by_fc=False, fit_by_cv=False, n_jobs=n_jobs)
  print(result_df_cv_ml)
  print(result_df_cv_ml.to_latex())


if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='Empirical evaluation')
  parser.add_argument('--mode', default='normal',
                      help='mode of empirical evaluation evaluation')
  parser.add_argument('--model', default=None,
                      help='model for which to run empirical evaluation evaluation')
  parser.add_argument('--n_jobs', default=None,
                      help='specifies the maximum number of concurrent jobs')


  args = parser.parse_args()

  if args.mode == 'normal':
    run_benchmark_train_test()
  elif args.mode == 'cv':
    run_benchmark_train_test_fit_by_cv(args.model)
  elif args.mode == 'cv_ml':
    run_benchmark_train_test_cv_ml()
  else:
    raise NotImplementedError()