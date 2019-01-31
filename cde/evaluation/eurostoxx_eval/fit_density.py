from cde.density_estimator.MDN import MixtureDensityNetwork
from cde.evaluation.eurostoxx_eval.load_dataset import make_overall_eurostoxx_df, target_feature_split

from sklearn.decomposition import PCA
import numpy as np

PCA_FEATURES = False

def pca_comp(X, n_components=4):
  pca = PCA(n_components=n_components)
  return pca.fit_transform(X)

def pca_var_explained(X):
  pca = PCA()
  pca.fit(X)
  return np.cumsum(pca.explained_variance_ratio_)


def main():
  # load data
  df = make_overall_eurostoxx_df()
  X, Y = target_feature_split(df, 'log_ret_1', filter_nan=True)
  X, Y = np.array(X), np.array(Y)

  if PCA_FEATURES:
    X = pca_comp(X, n_components=4)


  ndim_x, ndim_y = X.shape[1], 1

  mdn = MixtureDensityNetwork('mdn_empirical_no_pca', ndim_x, ndim_y, n_centers=20, n_training_epochs=500,
                              random_seed=22, x_noise_std=0.2, y_noise_std=0.1)
  mdn.fit(X,Y)

  mdn.plot2d(x_cond=[np.mean(X, axis=0)], ylim=(-0.02, 0.02))

if __name__ == '__main__':
  main()
