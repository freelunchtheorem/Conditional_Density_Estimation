from cde.density_estimator import KernelMixtureNetwork, ConditionalKernelDensityEstimation, MixtureDensityNetwork, \
  NeighborKernelDensityEstimation, LSConditionalDensityEstimation
from cde.empirical_evaluation.load_dataset import make_overall_eurostoxx_df, target_feature_split

from sklearn.model_selection import cross_validate
from cde.density_estimator import LSConditionalDensityEstimation, KernelMixtureNetwork, MixtureDensityNetwork, ConditionalKernelDensityEstimation, NeighborKernelDensityEstimation

from cde.evaluation.ConfigRunner import _create_configurations
import numpy as np
import time
import pandas as pd
import copy
from collections import OrderedDict

VALIDATION_PORTION = 0.2


FIT_BY_CV = True  # gridsearch # train/test split --> fit + evaluate --> nur ein train/test split
EVALUATE_BY_CV = False #

ndim_x = 14
ndim_y = 1

N_SAMPLES = 10**5
N_SEEDS = 5

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
  std_realized = np.sqrt(df_valid['RealizedVariation'][1:])

  # fit density model
  if eval_by_fc and not fit_by_cv:
    raise NotImplementedError
    # todo: implement eval by cv with n_folds different train_valid_splits, average over mean log, rmse_mu, rmse_std
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
    assert std_realized.shape == std_realized.shape

    # compute RMSE
    mu_rmse = np.sqrt(np.mean((mu_realized - mu_predicted) ** 2))
    std_rmse = np.sqrt(np.mean((std_realized - std_predicted) ** 2))
  else:
    mu_rmse, std_rmse = None, None

  return mean_logli, mu_rmse, std_rmse


def empirical_benchmark(model_dict, moment_r2=True, eval_by_fc=True, fit_by_cv=False):
  result_dict = {}

  for model_name, model in model_dict.items():
    print("Running likelihood fit and validation for %s" % model_name)
    for configuration_list in model:
      t = time.time()
      mean_logli_list = []
      mu_rmse_list = []
      std_rmse_list = []

      for seed_i in configuration_list:
        mean_logli, mu_rmse, std_rmse = empirical_evaluation(seed_i, VALIDATION_PORTION, moment_r2=moment_r2, eval_by_fc=eval_by_fc, fit_by_cv=fit_by_cv)

        mean_logli_list.append(mean_logli)
        mu_rmse_list.append(mu_rmse)
        std_rmse_list.append(std_rmse)

      mean_logli = np.mean(mean_logli_list)
      mu_rmse = np.mean(mu_rmse_list)
      std_rmse = np.mean(std_rmse_list)

      result_dict[str(configuration_list[0])] = mean_logli, mu_rmse, std_rmse
      print('%s results:' % model_name, result_dict[model_name])
      print('Duration of %s:' % model_name, time.time() - t)

  df = pd.DataFrame.from_dict(result_dict, 'index')
  df.columns = ['log_likelihood', 'rmse_mean', 'rmse_std']
  return df


def _add_seeds_to_est_params(n_seeds, configs):
  """ copies the configurations n_seeds times and adds seed numbers"""
  seeds = [20 + i for i in range(n_seeds)]
  for est_key in configs.keys():
    config_list = []
    for cfg in configs[est_key]:
      cfg_list = [copy.copy(cfg) for _ in range(n_seeds)]
      for i, cfg_i in enumerate(cfg_list):
        cfg_i['random_seed'] = seeds[i]

      config_list.append(cfg_list)
    configs[est_key] = config_list

  return configs


def create_seeds_model_dict(model_dict, verbose=False):
  """ duplicate model configs and assign seeds """
  configs = _create_configurations(model_dict)
  configs_w_seeds = _add_seeds_to_est_params(N_SEEDS, configs)

  """ initialize models """
  for estimator_name, estimator_params in configs_w_seeds.items():
    for seed_i, estimator_pack in enumerate(estimator_params):
      for cfg_variation_j, estimator in enumerate(estimator_pack):
        est_name = estimator_name + "_" + str(seed_i) + "_" + str(cfg_variation_j)
        if verbose: print("instantiating ", est_name)
        estimator["name"] = est_name
        configs_w_seeds[estimator_name][seed_i][cfg_variation_j] = globals()[estimator_name](**estimator)

  return configs_w_seeds


if __name__ == '__main__':

  if EVALUATE_BY_CV and not FIT_BY_CV:
    print("Evaluating estimators by CV")
  elif EVALUATE_BY_CV and FIT_BY_CV:
    print("Evaluating & fitting estimators by CV")
  elif not EVALUATE_BY_CV and FIT_BY_CV:
    print("Fitting estimators by CV for model selection")

  model_dict = {
    'ConditionalKernelDensityEstimation': {'ndim_x': [ndim_x], 'ndim_y': [ndim_y], 'bandwidth': ['normal_reference'] if not FIT_BY_CV else ['cv_ml'],
                                           'random_seed': [None]},
    'LSConditionalDensityEstimation': {'ndim_x': [ndim_x], 'ndim_y': [ndim_y], 'random_seed': [None]},
    'NeighborKernelDensityEstimation': {'ndim_x': [ndim_x], 'ndim_y': [ndim_y], 'random_seed': [None]},

    'MixtureDensityNetwork': {'name': ['MDN'], 'ndim_x': [ndim_x], 'ndim_y': [ndim_y], 'n_centers': [20], 'n_training_epochs': [200],
                              'x_noise_std': [0.2, None], 'y_noise_std': [0.1, None], 'random_seed': [None]
                              },
    'KernelMixtureNetwork': {'name': ['KMN'], 'ndim_x': [ndim_x], 'ndim_y': [ndim_y], 'n_centers': [50], 'n_training_epochs': [200],
                            'init_scales': [[0.7, 0.3]], 'x_noise_std': [0.2, None], 'y_noise_std': [0.1, None], 'random_seed': [None]
                             }
  }

  model_dict = create_seeds_model_dict(model_dict, verbose=VERBOSE)
  model_dict = OrderedDict(list(model_dict.items()))

  result_df = empirical_benchmark(model_dict, moment_r2=True, eval_by_fc=EVALUATE_BY_CV, fit_by_cv=FIT_BY_CV)
  print(result_df.to_latex())
  print(result_df)
