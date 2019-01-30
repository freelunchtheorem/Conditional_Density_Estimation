from ml_logger import logger

from cde.model_fitting.GoodnessOfFitResults import GoodnessOfFitResults
from cde.evaluation.simulation_eval import base_experiment
import cde.model_fitting.ConfigRunner as ConfigRunner
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.pyplot import cm
import pandas as pd
import copy


EXP_PREFIX = 'hyperparam_sweep'
RESULTS_FILE = 'results.pkl'

logger.configure(
  '/home/jonasrothfuss/Dropbox/Eigene_Dateien/Uni/WS17_18/Density_Estimation/Nonparametric_Density_Estimation/data/cluster',
  EXP_PREFIX)

results_from_pkl_file = dict(logger.load_pkl_log(RESULTS_FILE))
gof_result = GoodnessOfFitResults(single_results_dict=results_from_pkl_file)
results_df = gof_result.generate_results_dataframe(base_experiment.KEYS_OF_INTEREST + ['data_normalization', 'init_scales'])



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


""" Data Normalization vs no data normalization"""
plot_dict = dict([(simulator,
                   {"MDN_normalized": {"simulator": simulator, "estimator": "MixtureDensityNetwork", 'data_normalization': True},
                     "KMN_normalized": {"simulator": simulator, "estimator": "KernelMixtureNetwork", 'data_normalization': True},
                     "MDN_unnormalized": {"simulator": simulator, "estimator": "MixtureDensityNetwork", 'data_normalization': False},
                     "KMN_unnormalized": {"simulator": simulator, "estimator": "KernelMixtureNetwork", 'data_normalization': False},
                    }) for simulator in ["EconDensity", "ArmaJump", "SkewNormal"]])
fig = gof_result.plot_metric(plot_dict, metric="hellinger_distance", figsize=(16, 6))

fig.axes[0].set_title('EconDensity (std ~= 1)')
fig.axes[1].set_title('ArmaJump (std ~= 0.08)')
fig.axes[2].set_title('SkewNormal (std ~= 0.05)')

plt.tight_layout(w_pad=1)
plt.savefig(EXP_PREFIX + "/" + "data_normalization.png")

""" N-Centers"""
plot_dict = dict([(simulator,
                   {"MDN_10": {"simulator": simulator, "estimator": "MixtureDensityNetwork", 'n_centers': 10, 'data_normalization': True},
                    "MDN_20": {"simulator": simulator, "estimator": "MixtureDensityNetwork", 'n_centers': 20, 'data_normalization': True},
                    "KMN_50": {"simulator": simulator, "estimator": "KernelMixtureNetwork", 'n_centers': 50, 'data_normalization': True},
                    "KMN_200": {"simulator": simulator, "estimator": "KernelMixtureNetwork", 'n_centers': 200, 'data_normalization': True},
                    }) for simulator in ["EconDensity", "ArmaJump", "SkewNormal"]])
fig = gof_result.plot_metric(plot_dict, metric="hellinger_distance", figsize=(16, 6))

fig.axes[0].set_title('EconDensity')
fig.axes[1].set_title('ArmaJump')
fig.axes[2].set_title('SkewNormal')

plt.tight_layout(w_pad=1)
plt.savefig(EXP_PREFIX + "/" + "n_centers.png")

""" KMN Init Scales"""
gof_result_kmn = copy.deepcopy(gof_result)

results_df = results_df[results_df['estimator'] == 'KernelMixtureNetwork']
results_df['n_init_scales'] = [len(scales.split(',')) for scales in results_df['init_scales']]
gof_result_kmn.results_df = results_df


plot_dict = dict([(simulator,
                   {"KMN 2 scales": {"simulator": simulator, "estimator": "KernelMixtureNetwork", 'n_centers': 50, 'data_normalization': True, 'n_init_scales': 2},
                    "KMN 3 scales": {"simulator": simulator, "estimator": "KernelMixtureNetwork", 'n_centers': 50, 'data_normalization': True, 'n_init_scales': 3},
                    }) for simulator in ["EconDensity", "ArmaJump", "SkewNormal"]])
fig = gof_result_kmn.plot_metric(plot_dict, metric="hellinger_distance", figsize=(16, 6))

fig.axes[0].set_title('EconDensity')
fig.axes[1].set_title('ArmaJump')
fig.axes[2].set_title('SkewNormal')

plt.tight_layout(w_pad=1)
plt.savefig(EXP_PREFIX + "/" + "init_scales.png")

