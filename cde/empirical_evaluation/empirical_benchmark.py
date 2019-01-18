from cde.density_estimator import KernelMixtureNetwork, ConditionalKernelDensityEstimation, MixtureDensityNetwork, \
  NeighborKernelDensityEstimation, LSConditionalDensityEstimation
from cde.empirical_evaluation.load_dataset import make_overall_eurostoxx_df, target_feature_split

import numpy as np
import time
import pandas as pd
from collections import OrderedDict

VALIDATION_PORTION = 0.2

ndim_x = 14
ndim_y = 1

N_SAMPLES = 10**5

def train_valid_split(valid_portion):
  assert 0 < valid_portion < 1
  df = make_overall_eurostoxx_df()
  df = df.dropna()

  # split data into train and validation set
  split_index = int(df.shape[0] * (1.0 - valid_portion))

  df_train = df[:split_index]
  df_valid = df[split_index:]

  return df_train, df_valid


def empirical_evaluation(estimator, valid_portion=0.2, moment_r2=True):
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
  estimator.fit(X_train, Y_train)

  # compute avg. log likelihood
  mean_logli = np.mean(np.log(estimator.pdf(X_valid, Y_valid)))

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

def empirical_benchmark(model_dict, moment_r2=True):
  result_dict = {}

  for model_name, model in model_dict.items():
    print("Running likelihood fit and validation for %s"%model_name)
    t = time.time()
    result_dict[model_name] = empirical_evaluation(model, VALIDATION_PORTION, moment_r2=moment_r2)
    print('%s results:' % model_name, result_dict[model_name])
    print('Duration of %s:'%model_name, time.time() - t)

  df = pd.DataFrame.from_dict(result_dict, 'index')
  df.columns = ['log_likelihood', 'rmse_mean', 'rmse_std']
  return df

if __name__ == '__main__':
  model_dict = {
    'MDN w/ noise smoothing:': MixtureDensityNetwork('mdn1', ndim_x, ndim_y, n_centers=20, n_training_epochs=2000,
                                random_seed=22, x_noise_std=0.2, y_noise_std=0.1),
    'MDN w/o noise smoothing': MixtureDensityNetwork('mdn2', ndim_x, ndim_y, n_centers=20,
                                n_training_epochs=2000, random_seed=22, x_noise_std=None, y_noise_std=None),

    'KMN w/ noise smoothing': KernelMixtureNetwork('kmn1', ndim_x, ndim_y, n_centers=50, n_training_epochs=2000, init_scales=[0.7, 0.3],
                                random_seed=22, x_noise_std=0.2, y_noise_std=0.1),
    'KMN w/o noise smoothing':  KernelMixtureNetwork('kmn2', ndim_x, ndim_y, n_centers=50, n_training_epochs=2000, init_scales=[0.7, 0.3],
                                random_seed=22, x_noise_std=None, y_noise_std=None),
    'NKDE': NeighborKernelDensityEstimation('NKDE', ndim_x, ndim_y),

    'LSCDE': LSConditionalDensityEstimation('CKDE', ndim_x, ndim_y),
    'CKDE normal_reference': ConditionalKernelDensityEstimation('ckde', ndim_x, ndim_y,
                                                                bandwidth='normal_reference'),
    'CKDE cv_ml': ConditionalKernelDensityEstimation('ckde', ndim_x, ndim_y,
                                                                bandwidth='cv_ml'),
  }
  model_dict = OrderedDict(list(model_dict.items()))

  result_df = empirical_benchmark(model_dict, moment_r2=True)
  print(result_df.to_latex())
  print(result_df)
