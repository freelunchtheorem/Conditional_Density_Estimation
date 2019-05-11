from ml_logger import logger

from cde.model_fitting.GoodnessOfFitResults import GoodnessOfFitResults
from cde.evaluation.simulation_eval import base_experiment
import cde.model_fitting.ConfigRunner as ConfigRunner
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle


EXP_PREFIX = 'question1_noise_reg_x_v1'
RESULTS_FILE = 'results.pkl'
LOCATION = '/home/simon/Documents/KIT/Informatik/Bachelorarbeit/Conditional_Density_Estimation/data/cluster/{}/{}'.format(
    EXP_PREFIX,
    RESULTS_FILE
)


logger.configure(
    '/home/simon/Documents/KIT/Informatik/Bachelorarbeit/Conditional_Density_Estimation/data/cluster',
    EXP_PREFIX)

results_from_pkl_file = dict(logger.load_pkl_log(RESULTS_FILE))
gof_result = GoodnessOfFitResults(single_results_dict=results_from_pkl_file)
results_df = gof_result.generate_results_dataframe(base_experiment.KEYS_OF_INTEREST)
# Fix problem with accessing columns indexed by None values
results_df.replace(to_replace=[None], value='None', inplace=True)

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
estimator = 'NormalizingFlowEstimator'
title = "%s - Y-Noise Regularization"%(estimator,)
plot_dict = dict([(simulator,
   {
     "x_noise_std=0.4": {"estimator": estimator, "x_noise_std": 0.4, "y_noise_std": 'None', "simulator": simulator},
     "x_noise_std=0.2": {"estimator": estimator, "x_noise_std": 0.2, "y_noise_std": 'None', "simulator": simulator},
     "x_noise_std=0.1": {"estimator": estimator, "x_noise_std": 0.1, "y_noise_std": 'None', "simulator": simulator},
     "x_noise_std=0.0": {"estimator": estimator, "x_noise_std": 'None', "y_noise_std": 'None', "simulator": simulator}
   }) for simulator in ["EconDensity", "ArmaJump", "GaussianMixture", "SkewNormal"]])

fig = gof_result.plot_metric(plot_dict, metric="hellinger_distance", figsize=(14,10), layout=(2,2))
plt.suptitle(title, fontsize=TITLE_SIZE)
plt.tight_layout(h_pad=2, rect=[0, 0, 1, 0.95])
plt.savefig(EXP_PREFIX + "/" +"%s_x_noise.png"%(estimator,))
plt.clf()

""" Y-Noise Regularization"""
estimator = 'NormalizingFlowEstimator'
title = "%s - Y-Noise Regularization"%(estimator,)
plot_dict = dict([(simulator,
  {
    "y_noise_std=0.2": {"estimator": estimator, "y_noise_std": 0.2, "x_noise_std": 'None', "simulator": simulator},
    "y_noise_std=0.1": {"estimator": estimator, "y_noise_std": 0.1, "x_noise_std": 'None', "simulator": simulator},
    "y_noise_std=0.02": {"estimator": estimator, "y_noise_std": 0.02, "x_noise_std": 'None',"simulator": simulator},
    "y_noise_std=0.0": {"estimator": estimator, "y_noise_std": 'None', "x_noise_std": 'None', "simulator": simulator}
  },
) for simulator in ["EconDensity", "ArmaJump", "GaussianMixture", "SkewNormal"]])

fig = gof_result.plot_metric(plot_dict, metric="hellinger_distance", figsize=(14,10), layout=(2,2))
plt.suptitle(title, fontsize=TITLE_SIZE)
plt.tight_layout(h_pad=2, rect=[0, 0, 1, 0.95])
plt.savefig(EXP_PREFIX + "/" +"%s_y_noise.png"%(estimator,))
plt.clf()

""" XY-Noise Regularization"""
color = iter(['red', 'orange', 'blue', 'green'])
title = "Effect of XY-Noise Regularization (x_noise_std=0.2, y_noise_std=0.1)"
plot_dict = dict([(simulator,
    {
      "MDN noise": {"estimator": "MixtureDensityNetwork", "y_noise_std": 0.1, "x_noise_std": 0.2, "n_centers": 10, "simulator": simulator},
      "MDN no noise": {"estimator": "MixtureDensityNetwork", "y_noise_std": 'None', "x_noise_std": 'None', "n_centers": 10, "simulator": simulator},
      "NF noise": {"estimator": "NormalizingFlowEstimator", "y_noise_std": 0.1, "x_noise_std": 0.2, "simulator": simulator},
      "NF no noise": {"estimator": "NormalizingFlowEstimator", "y_noise_std": 'None', "x_noise_std": 'None', "simulator": simulator}
    },
) for simulator in ["EconDensity", "ArmaJump", "SkewNormal"]])

fig = gof_result.plot_metric(plot_dict, metric="hellinger_distance", figsize=(14, 4.5), layout=(1, 3), color=color)
#plt.suptitle(title, fontsize=TITLE_SIZE)

for i, ax in enumerate(fig.axes):
  if i != 2:
    ax.get_legend().remove()
  ax.set_ylabel('Hellinger distance')
  ax.set_xlabel('number of training samples')
  ax.set_xticks([200, 500, 1000, 2000, 5000])
  ax.set_xticklabels([200, 500, 1000, 2000, 5000])


plt.tight_layout(h_pad=2, rect=[0, 0, 1, 1])
plt.savefig(EXP_PREFIX + "/" + "xy_noise_overview.png")
plt.savefig(EXP_PREFIX + "/" + "xy_noise_overview.pdf")
plt.clf()

""" X-Y Noise Reg heatplots"""
""" X-Y Noise Reg heatplots"""
n_samples = 1600
metric = 'hellinger_distance'
x_noise_vals = list(reversed(['None', 0.1, 0.2, 0.4]))
y_noise_vals = ['None', 0.01, 0.02, 0.05, 0.1, 0.2]
estimator = 'NormalizingFlowEstimator'
fig, axarr = plt.subplots(2, 2, figsize=(12, 7.5))
axarr = axarr.flatten()
for k, simulator in enumerate(["EconDensity", "ArmaJump", "GaussianMixture", "SkewNormal"]):
    result_grid = np.empty((len(x_noise_vals), len(y_noise_vals)))
    for i, x_noise_std in enumerate(x_noise_vals):
        for j, y_noise_std in enumerate(y_noise_vals):
            graph_dict = {"estimator": estimator, "x_noise_std": x_noise_std, "y_noise_std": y_noise_std,
                          "simulator": simulator, 'n_observations': n_samples}
            sub_df = results_df.loc[(results_df[list(graph_dict)] == pd.Series(graph_dict)).all(axis=1)]
            result_grid[i, j] = sub_df[metric].mean()
    im = axarr[k].imshow(result_grid)
    # annotate pixels
    for i, x_noise_std in enumerate(x_noise_vals):
        for j, y_noise_std in enumerate(y_noise_vals):
            axarr[k].text(j, i, "%.3f"%result_grid[i, j],
                          ha="center", va="center", color="w")
    axarr[k].set_ylabel("x-noise std")
    axarr[k].set_xlabel("y-noise std")
    axarr[k].set_yticks(np.arange(len(x_noise_vals)))
    axarr[k].set_xticks(np.arange(len(y_noise_vals)))
    axarr[k].set_yticklabels([str(val) for val in x_noise_vals])
    axarr[k].set_xticklabels([str(val) for val in y_noise_vals])
    cbar = axarr[k].figure.colorbar(im, ax=axarr[k], shrink=0.8)
    cbar.ax.set_ylabel("Hellinger distance", rotation=-90, va="bottom")
    axarr[k].set_title(simulator)
plt.tight_layout(w_pad=3.5, rect=[0, 0, 1, 1])
#plt.suptitle("Hellinger Distance X-Y-Noise:\n%s (%i centers) - %i observations"%(estimator, n_centers, n_samples), fontsize=TITLE_SIZE)
plt.savefig(EXP_PREFIX + "/" + "%s_%i_samples_xy_noise_heatmap.pdf" % (estimator, n_samples))
plt.savefig(EXP_PREFIX + "/" + "%s_%i_samples_xy_noise_heatmap.png" % (estimator, n_samples))
print('Finished plotting')