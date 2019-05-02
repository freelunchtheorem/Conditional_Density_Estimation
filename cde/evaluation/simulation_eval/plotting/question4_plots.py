from ml_logger import logger

from cde.model_fitting.GoodnessOfFitResults import GoodnessOfFitResults
from cde.evaluation.simulation_eval import base_experiment
import matplotlib.pyplot as plt

from matplotlib.pyplot import cm
from matplotlib.ticker import FormatStrFormatter, StrMethodFormatter
import pandas as pd
import os

SMALL_SIZE = 13
MEDIUM_SIZE = 14
LARGE_SIZE = 16
TITLE_SIZE = 20

LINEWIDTH = 6

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=LARGE_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize

## SKEW Normal

EXP_PREFIX = 'question4_benchmark_skew_NF'
RESULTS_FILE = 'results.pkl'

if not os.path.isdir(EXP_PREFIX):
    os.makedirs(EXP_PREFIX)

logger.configure(
  '/home/simon/Documents/KIT/Informatik/Bachelorarbeit/Conditional_Density_Estimation/data/cluster',
  EXP_PREFIX)

results_from_pkl_file = dict(logger.load_pkl_log(RESULTS_FILE))
gof_result = GoodnessOfFitResults(single_results_dict=results_from_pkl_file)
results_df = gof_result.generate_results_dataframe(base_experiment.KEYS_OF_INTEREST + ['bandwidth'])


plot_dict = dict([(simulator,
                   {"MDN": {"simulator": simulator, "estimator": "MixtureDensityNetwork", "x_noise_std": 0.1},
                     "KMN": {"simulator": simulator, "estimator": "KernelMixtureNetwork", "x_noise_std": 0.1},
                     "LSCDE": {"simulator": simulator, "estimator": "LSConditionalDensityEstimation"},
                     "CKDE": {"simulator": simulator, "estimator": "ConditionalKernelDensityEstimation", "bandwidth": "normal_reference"},
                     "CKDE_CV": {"simulator": simulator, "estimator": "ConditionalKernelDensityEstimation", "bandwidth": "cv_ml"},
                     "NKDE": {"simulator": simulator, "estimator": "NeighborKernelDensityEstimation"},
                    "NF": {"simulator": simulator, "estimator": "NormalizingFlowEstimator",}
                    }) for simulator in ["EconDensity", "ArmaJump", "SkewNormal"]])

fig = gof_result.plot_metric(plot_dict, metric="hellinger_distance", figsize=(15, 5.5))

## ARMA JUMP

EXP_PREFIX = 'question4_benchmark_arma_jump'

logger.configure(
  '/home/simon/Documents/KIT/Informatik/Bachelorarbeit/Conditional_Density_Estimation/data/cluster',
  EXP_PREFIX)

results_from_pkl_file = dict(logger.load_pkl_log(RESULTS_FILE))
gof_result = GoodnessOfFitResults(single_results_dict=results_from_pkl_file)
results_df = gof_result.generate_results_dataframe(base_experiment.KEYS_OF_INTEREST + ['bandwidth'])


plot_dict = dict([(simulator,
                   {"MDN": {"simulator": simulator, "estimator": "MixtureDensityNetwork", "x_noise_std": 0.1},
                    "KMN": {"simulator": simulator, "estimator": "KernelMixtureNetwork", "x_noise_std": 0.1},
                    "LSCDE": {"simulator": simulator, "estimator": "LSConditionalDensityEstimation"},
                    "CKDE": {"simulator": simulator, "estimator": "ConditionalKernelDensityEstimation", "bandwidth": "normal_reference"},
                    "CKDE_CV": {"simulator": simulator, "estimator": "ConditionalKernelDensityEstimation", "bandwidth": "cv_ml"},
                    "NKDE": {"simulator": simulator, "estimator": "NeighborKernelDensityEstimation"},
                    "NF": {"simulator": simulator, "estimator": "NormalizingFlowEstimator",}
                    }) for simulator in ["EconDensity", "ArmaJump", "SkewNormal",]])

#color = iter(cm.rainbow(np.linspace(0, 1, 7))[2:])
fig = gof_result.plot_metric(plot_dict, metric="hellinger_distance", fig=fig)



EXP_PREFIX = 'question4_benchmark_econ_density'

logger.configure(
    '/home/simon/Documents/KIT/Informatik/Bachelorarbeit/Conditional_Density_Estimation/data/cluster',
    EXP_PREFIX)

results_from_pkl_file = dict(logger.load_pkl_log(RESULTS_FILE))
gof_result = GoodnessOfFitResults(single_results_dict=results_from_pkl_file)
results_df = gof_result.generate_results_dataframe(base_experiment.KEYS_OF_INTEREST + ['bandwidth'])


plot_dict = dict([(simulator,
                   {"MDN": {"simulator": simulator, "estimator": "MixtureDensityNetwork", "x_noise_std": 0.1},
                    "KMN": {"simulator": simulator, "estimator": "KernelMixtureNetwork", "x_noise_std": 0.1},
                    "LSCDE": {"simulator": simulator, "estimator": "LSConditionalDensityEstimation"},
                    "CKDE": {"simulator": simulator, "estimator": "ConditionalKernelDensityEstimation", "bandwidth": "normal_reference"},
                    "CKDE_CV": {"simulator": simulator, "estimator": "ConditionalKernelDensityEstimation", "bandwidth": "cv_ml"},
                    "NKDE": {"simulator": simulator, "estimator": "NeighborKernelDensityEstimation"},
                    "NF": {"simulator": simulator, "estimator": "NormalizingFlowEstimator",}
                    }) for simulator in ["EconDensity", "ArmaJump", "SkewNormal",]])

#color = iter(cm.rainbow(np.linspace(0, 1, 7))[2:])
fig = gof_result.plot_metric(plot_dict, metric="hellinger_distance", fig=fig)

plt.tight_layout(h_pad=2, rect=[0, 0.0, 1, 1])

# legend
for i, ax in enumerate(fig.axes):
  ax.get_legend().remove()
  ax.set_ylabel('Hellinger distance')
  ax.set_xlabel('number of training samples')
  ax.set_xticks([500, 1000, 2000, 5000])
  ax.set_xticklabels([500, 1000, 2000, 5000])

fig.legend(['MDN', 'KMN', 'LSCDE', 'CKDE', 'CKDE-CV', 'NKDE', 'NF'], ncol=7, loc='lower center')

if not os.path.isdir(EXP_PREFIX):
    os.makedirs(EXP_PREFIX)

fig.savefig(os.path.join(EXP_PREFIX, "simulation_benchmark.pdf"))
fig.savefig(os.path.join(EXP_PREFIX, "simulation_benchmark.png"))




