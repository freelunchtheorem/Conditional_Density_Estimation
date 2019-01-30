from cde.density_estimator.MDN import MixtureDensityNetwork
from cde.evaluation.eurostoxx_eval.load_dataset import make_overall_eurostoxx_df, target_feature_split

import numpy as np
import os
from matplotlib import pyplot as plt

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data'))

def main():
  # load data
  df = make_overall_eurostoxx_df()
  X, Y, features = target_feature_split(df, 'log_ret_1', filter_nan=True, return_features=True)
  X, Y = np.array(X), np.array(Y)

  ndim_x, ndim_y = X.shape[1], 1

  mdn = MixtureDensityNetwork('mdn_empirical_no_pca', ndim_x, ndim_y, n_centers=20, n_training_epochs=10,
                              random_seed=28, x_noise_std=0.2, y_noise_std=0.1)
  mdn.fit(X,Y)

  X_mean = np.mean(X, axis=0)
  X_std = np.mean(X, axis=0)

  # individual plots
  for i, feature in enumerate(features):
    factor = np.zeros(X_std.shape)
    factor[i] = X_std[i]
    x_cond = [X_mean-2*factor, X_mean-1*factor, X_mean, X_mean+1*factor, X_mean+2*factor]
    mdn.plot2d(x_cond=x_cond, ylim=(-0.04, 0.04), show=False)
    plt.legend(['mean-2*std', 'mean-1*std', 'mean', 'mean+1*std', 'mean+2*std'])
    plt.title(feature)
    fig_path = os.path.join(DATA_DIR, 'plots/feature_selection/' + feature + '.png')
    plt.savefig(fig_path)


  # one large plot
  resolution = 100
  ncols = 3
  fig, axes = plt.subplots(nrows=5, ncols=ncols, figsize=(12, 16))
  y = np.linspace(-0.04, 0.04, resolution)

  n = 0
  for i, feature in enumerate(features):
    if n == 2:
      n+=1
    factor = np.zeros(X_std.shape)
    factor[i] = X_std[i]
    x_cond = [X_mean + 2 * factor, X_mean + 1 * factor, X_mean, X_mean - 1 * factor, X_mean - 2 * factor]

    for j in range(len(x_cond)):
      x = np.array([x_cond[j] for _ in range(resolution)])
      z = mdn.pdf(x, y)
      axes[n // ncols][n % ncols].plot(y, z)

    axes[n // ncols][n % ncols].set_title(feature)
    n += 1

  axes[0][0].set_xlabel('log return')
  axes[0][0].set_ylabel('probability density log-returns')

  # make top right plot disappear
  axes[0, 2].tick_params(colors='white')
  for spine in axes[0, 2].spines.values(): spine.set_color('white')

  fig.legend(['mean+2*std', 'mean+1*std', 'mean', 'mean-1*std', 'mean-2*std'], loc=(.77, 0.88))
  fig.tight_layout()

  fig_path = os.path.join(DATA_DIR, 'plots/feature_selection/feature_selection_all_variables.png')
  fig.savefig(fig_path)

  fig_path = os.path.join(DATA_DIR, 'plots/feature_selection/feature_selection_all_variables.pdf')
  fig.savefig(fig_path)




if __name__ == '__main__':
  main()
