from cde.density_estimator import KernelMixtureNetwork, ConditionalKernelDensityEstimation, MixtureDensityNetwork
from cde.empirical_evaluation.load_dataset import make_overall_eurostoxx_df, target_feature_split

import numpy as np

VALIDATION_PORTION = 0.2

ndim_x = 14
ndim_y = 1


def train_valid_split(valid_portion):
  assert 0 < valid_portion < 1
  df = make_overall_eurostoxx_df()
  df = df.dropna()

  # split data into train and validation set
  split_index = int(df.shape[0] * (1.0 - valid_portion))

  df_train = df[:split_index]
  df_valid = df[split_index:]

  return df_train, df_valid

def compute_r2_moments(estimator, valid_portion=0.2):
  """
  Fits the estimator and, based on a left out validation splot, computes the
  Root Mean Squared Error (RMSE) between realized and estimated mean and std

  Args:
    estimator: estimator object
    valid_portion: portion of dataset to be separated as validation set

  Returns:
    (mu_rmse, std_rmse): RMSE between realized and estimated mean and std
  """

  # get data and split into train and valid set
  df_train, df_valid = train_valid_split(valid_portion)

  X_train, Y_train = target_feature_split(df_train, 'log_ret_1', filter_nan=True)
  X_valid, _ = target_feature_split(df_valid, 'log_ret_1', filter_nan=True)


  # realized moments
  mu_realized = df_valid['log_ret_last_period'][1:]
  std_realized = np.sqrt(df_valid['RealizedVariation'][1:])

  # fit density model
  estimator.fit(X_train, Y_train)

  # predict mean and std
  mu_predicted = estimator.mean_(X_valid).flatten()[:-1]
  std_predicted = np.sqrt(estimator.covariance(X_valid).flatten())[:-1]

  assert mu_realized.shape == mu_predicted.shape
  assert std_realized.shape == std_realized.shape

  # compute RMSE
  mu_rmse = np.sqrt(np.mean((mu_realized - mu_predicted) ** 2))
  std_rmse = np.sqrt(np.mean((std_realized - std_predicted) ** 2))

  return mu_rmse, std_rmse

def compute_valid_logli(estimator, valid_portion=0.2):
  """
  Fits the estimator on the train split and computes the mean log-likelihood on the validation set

   Args:
     estimator: estimator object
     valid_portion: portion of dataset to be separated as validation set

   Returns:
     Average log_likelihood in the validation set
   """

  # get data and split into train and valid set
  df_train, df_valid = train_valid_split(valid_portion)

  X_train, Y_train = target_feature_split(df_train, 'log_ret_1', filter_nan=True)
  X_valid, Y_valid = target_feature_split(df_valid, 'log_ret_1', filter_nan=True)

  # fit density model
  estimator.fit(X_train, Y_train)

  # compute avg. log likelihood
  mean_logli = np.mean(np.log(estimator.pdf(X_valid, Y_valid)))
  return mean_logli

def logli_benchmark(model_dict):
  result_dict = {}

  for model_name, model in model_dict.items():
    result_dict[model_name] = compute_valid_logli(model, VALIDATION_PORTION)

  return result_dict

def r2_benchmark(model_dict):
  result_dict = {}

  for model_name, model in model_dict.items():
    result_dict[model_name] = compute_r2_moments(model, VALIDATION_PORTION)

  return result_dict

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
    'CKDE normal_reference': ConditionalKernelDensityEstimation('ckde', ndim_x, ndim_y,
                                                                bandwidth_selection='normal_reference'),
    'CKDE cv_ml': ConditionalKernelDensityEstimation('ckde', ndim_x, ndim_y,
                                                                bandwidth_selection='cv_ml'),
    'CKDE cv_ls': ConditionalKernelDensityEstimation('ckde', ndim_x, ndim_y,
                                                     bandwidth_selection='cv_ls'),
  }

  result_dict = logli_benchmark(model_dict)
  from pprint import pprint
  pprint(result_dict)
