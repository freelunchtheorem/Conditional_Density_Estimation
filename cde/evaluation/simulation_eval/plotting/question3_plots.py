from ml_logger import logger

from cde.model_fitting.GoodnessOfFitResults import GoodnessOfFitResults
from cde.evaluation.simulation_eval import base_experiment
import cde.model_fitting.ConfigRunner as ConfigRunner
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.pyplot import cm
import pandas as pd


EXP_PREFIX = 'question3_KDE'
RESULTS_FILE = 'results.pkl'

logger.configure(
  '/home/jonasrothfuss/Dropbox/Eigene_Dateien/Uni/WS17_18/Density_Estimation/Nonparametric_Density_Estimation/data/cluster',
  EXP_PREFIX)

results_from_pkl_file = dict(logger.load_pkl_log(RESULTS_FILE))
gof_result = GoodnessOfFitResults(single_results_dict=results_from_pkl_file)
results_df = gof_result.generate_results_dataframe(base_experiment.KEYS_OF_INTEREST + ['bandwidth_selection'])

#gof_result = ConfigRunner.load_dumped_estimators(gof_result, task_id=[5])


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


title = "KDE vs NN-based"
color = iter(cm.rainbow(np.linspace(0, 1, 6))[:2])
plot_dict = dict([(simulator,
                   {"KDE-normal-reference": {"simulator": simulator, "estimator": "ConditionalKernelDensityEstimation", "bandwidth_selection": "normal_reference"},
                     "KDE-ml-cv": {"simulator": simulator, "estimator": "ConditionalKernelDensityEstimation", "bandwidth_selection": "cv_ml"}
                    }) for simulator in ["EconDensity", "ArmaJump", "SkewNormal"]])

fig = gof_result.plot_metric(plot_dict, metric="js_divergence", figsize=(14, 5), layout=(1, 3), color=color)
plt.suptitle(title, fontsize=TITLE_SIZE)
plt.tight_layout(h_pad=2, rect=[0, 0, 1, 0.95])

# add KMN and MDN plots
logger.configure(
  '/home/jonasrothfuss/Dropbox/Eigene_Dateien/Uni/WS17_18/Density_Estimation/Nonparametric_Density_Estimation/data/cluster',
  "question1_noise_reg_x_v1")

results_from_pkl_file = dict(logger.load_pkl_log(RESULTS_FILE))
gof_result = GoodnessOfFitResults(single_results_dict=results_from_pkl_file)
_ = gof_result.generate_results_dataframe(base_experiment.KEYS_OF_INTEREST)

plot_dict = dict([(simulator,
                   {"MDN": {"simulator": simulator, "estimator": "MixtureDensityNetwork", "x_noise_std": 0.2, "y_noise_std": 0.1, "n_centers": 10},
                     "KMN": {"simulator": simulator, "estimator": "KernelMixtureNetwork", "x_noise_std": 0.2, "y_noise_std": 0.1, "n_centers": 20},
                    }) for simulator in ["EconDensity", "ArmaJump", "SkewNormal"]])

color = iter(cm.rainbow(np.linspace(0, 1, 4))[2:])
fig = gof_result.plot_metric(plot_dict, metric="hellinger_distance", fig=fig, color=color)

plt.savefig("kde_vs_nn_based")
plt.clf()



