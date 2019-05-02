from ml_logger import logger

from cde.model_fitting.GoodnessOfFitResults import GoodnessOfFitResults
from cde.evaluation.simulation_eval import base_experiment
import cde.model_fitting.ConfigRunner as ConfigRunner
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

EXP_PREFIX = 'question1_noise_reg_x_v1'
RESULTS_FILE = 'results.pkl'

logger.configure(
  '/home/simon/Documents/KIT/Informatik/Bachelorarbeit/Conditional_Density_Estimation/data/cluster',
  EXP_PREFIX)

if not os.path.isdir(EXP_PREFIX):
  os.makedirs(EXP_PREFIX)

results_from_pkl_file = dict(logger.load_pkl_log(RESULTS_FILE))
gof_result = GoodnessOfFitResults(single_results_dict=results_from_pkl_file)
results_df = gof_result.generate_results_dataframe(base_experiment.KEYS_OF_INTEREST)

#gof_result = ConfigRunner.load_dumped_estimators(gof_result)


SMALL_SIZE = 11
MEDIUM_SIZE = 12
LARGE_SIZE = 16
TITLE_SIZE = 20

LINEWIDTH = 6

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=LARGE_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize


""" X-Noise Regularization"""
for estimator, n_centers in [("MixtureDensityNetwork", 10), ("KernelMixtureNetwork", 20)]:
  title = "%s (%i kernels) - X-Noise Regularization"%(estimator, n_centers)
  plot_dict = dict([(simulator,
       {
         "x_noise_std=0.4": {"estimator": estimator, "x_noise_std": 0.4, "y_noise_std": None, "n_centers": n_centers, "simulator": simulator},
         "x_noise_std=0.2": {"estimator": estimator, "x_noise_std": 0.2, "y_noise_std": None, "n_centers": n_centers, "simulator": simulator},
         "x_noise_std=0.1": {"estimator": estimator, "x_noise_std": 0.1, "y_noise_std": None, "n_centers": n_centers, "simulator": simulator},
         "x_noise_std=0.0": {"estimator": estimator, "x_noise_std": None, "y_noise_std": None, "n_centers": n_centers, "simulator": simulator}
       }) for simulator in ["EconDensity", "ArmaJump", "GaussianMixture", "SkewNormal"]])

  fig = gof_result.plot_metric(plot_dict, metric="js_divergence", figsize=(14,10), layout=(2,2))
  plt.suptitle(title, fontsize=TITLE_SIZE)
  plt.tight_layout(h_pad=2, rect=[0, 0, 1, 0.95])
  plt.savefig(os.path.join(EXP_PREFIX,"%s_%i_x_noise.png"%(estimator, n_centers)))
  plt.clf()

""" Y-Noise Regularization"""
for estimator, n_centers in [("MixtureDensityNetwork", 10), ("KernelMixtureNetwork", 20)]:
  title = "%s (%i kernels) - Y-Noise Regularization"%(estimator, n_centers)
  plot_dict = dict([(simulator,
      {
        "y_noise_std=0.2": {"estimator": estimator, "y_noise_std": 0.2, "x_noise_std": None, "n_centers": n_centers, "simulator": simulator},
        "y_noise_std=0.1": {"estimator": estimator, "y_noise_std": 0.1, "x_noise_std": None, "n_centers": n_centers, "simulator": simulator},
        "y_noise_std=0.05": {"estimator": estimator, "y_noise_std": 0.05, "x_noise_std": None, "n_centers": n_centers, "simulator": simulator},
        "y_noise_std=0.02": {"estimator": estimator, "y_noise_std": 0.02, "x_noise_std": None, "n_centers": n_centers, "simulator": simulator},
        "y_noise_std=0.01": {"estimator": estimator, "y_noise_std": 0.01, "x_noise_std": None, "n_centers": n_centers, "simulator": simulator},
        "y_noise_std=0.0": {"estimator": estimator, "y_noise_std": None, "x_noise_std": None, "n_centers": n_centers, "simulator": simulator}
      },
  ) for simulator in ["EconDensity", "ArmaJump", "GaussianMixture", "SkewNormal"]])

  fig = gof_result.plot_metric(plot_dict, metric="js_divergence", figsize=(14,10), layout=(2,2))
  plt.suptitle(title, fontsize=TITLE_SIZE)
  plt.tight_layout(h_pad=2, rect=[0, 0, 1, 0.95])
  plt.savefig(os.path.join(EXP_PREFIX,"%s_%i_y_noise.png"%(estimator, n_centers)))
  plt.clf()

""" X-Y Noise Reg heatplots"""
n_samples = 1600
metric = 'js_divergence'
x_noise_vals = [None, 0.1, 0.2, 0.4]
y_noise_vals = list(reversed([None, 0.01, 0.02, 0.05, 0.1, 0.2]))

for estimator, n_centers in [("MixtureDensityNetwork", 10), ("KernelMixtureNetwork", 20)]:
  fig, axarr = plt.subplots(2, 2, figsize=(10, 8))
  axarr = axarr.flatten()
  for k, simulator in enumerate(["EconDensity", "ArmaJump", "GaussianMixture", "SkewNormal"]):

    result_grid = np.empty((len(x_noise_vals), len(y_noise_vals)))

    for i, x_noise_std in enumerate(x_noise_vals):
      for j, y_noise_std in enumerate(y_noise_vals):

        graph_dict = {"estimator": estimator, "x_noise_std": x_noise_std, "y_noise_std": y_noise_std,
                        "n_centers": n_centers, "simulator": simulator, 'n_observations': n_samples}

        sub_df = results_df.loc[(results_df[list(graph_dict)] == pd.Series(graph_dict)).all(axis=1)]

        result_grid[i,j] = sub_df[metric].mean()

    im = axarr[k].imshow(result_grid)

    # annotate pixels
    for i, x_noise_std in enumerate(x_noise_vals):
      for j, y_noise_std in enumerate(y_noise_vals):
        axarr[k].text(j, i, "%.3f"%result_grid[i, j],
                      ha="center", va="center", color="w")

    axarr[k].set_xlabel("x_noise_std")
    axarr[k].set_ylabel("y_noise_std")
    axarr[k].set_xticks(np.arange(len(x_noise_vals)))
    axarr[k].set_yticks(np.arange(len(y_noise_vals)))
    axarr[k].set_xticklabels([str(val) for val in x_noise_vals])
    axarr[k].set_yticklabels([str(val) for val in y_noise_vals])
    cbar = axarr[k].figure.colorbar(im, ax=axarr[k], shrink=0.9)
    cbar.ax.set_ylabel("JS-divergence", rotation=-90, va="bottom")
    axarr[k].set_title(simulator)

  plt.tight_layout(w_pad=2, rect=[0, 0, 1, 0.9])
  plt.suptitle("JS-Divergence X-Y-Noise:\n%s (%i centers) - %i observations"%(estimator, n_centers, n_samples), fontsize=TITLE_SIZE)
  plt.savefig(os.path.join(EXP_PREFIX, "%s_%i_%iobs_xy_noise_heatmap.png" % (estimator, n_centers, n_samples)))