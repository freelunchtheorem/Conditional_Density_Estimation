from ml_logger import logger

from cde.model_fitting.GoodnessOfFitResults import GoodnessOfFitResults
from cde.evaluation.simulation_eval import base_experiment
import cde.model_fitting.ConfigRunner as ConfigRunner
import matplotlib.pyplot as plt
import os

EXP_PREFIX = "question6_noise_schedules"
RESULTS_FILE = "results.pkl"
CLUSTER_DIR = "/local/rojonas/cde/data/local"
LOCATION = "{}/{}/{}".format(CLUSTER_DIR, EXP_PREFIX, RESULTS_FILE)
DATA_DIR_LOCAL = "/home/jonasrothfuss/Dropbox/Eigene_Dateien/ETH/02_Projects/02_Noise_Regularization/02_Code_Conditional_Density_Estimation/data/cluster"

logger.configure(
    #"/local/rojonas/cde/data/local",
    DATA_DIR_LOCAL,
    EXP_PREFIX,
)

results_from_pkl_file = dict(logger.load_pkl_log(RESULTS_FILE))
gof_result = GoodnessOfFitResults(single_results_dict=results_from_pkl_file)
results_df = gof_result.generate_results_dataframe(base_experiment.KEYS_OF_INTEREST_LOGPROB)
results_df.replace(to_replace=[None], value="None", inplace=True)

estimators = [
    "MixtureDensityNetwork",
    "KernelMixtureNetwork",
    "NormalizingFlowEstimator"
]
simulators = ["EconDensity", "GaussianMixture", "SkewNormal"]

# rules of thumb
for estimator in estimators:
    plot_dict = dict(
        [
            (
                simulator,
                {
                    "rule_of_thumb_1.0": {
                        "simulator": simulator,
                        "estimator": estimator,
                        "adaptive_noise_fn": "rule_of_thumb_1.00"
                    },
                    "rule_of_thumb_0.7": {
                        "simulator": simulator,
                        "estimator": estimator,
                        "adaptive_noise_fn": "rule_of_thumb_0.70"
                    },
                    "rule_of_thumb_0.5": {
                        "simulator": simulator,
                        "estimator": estimator,
                        "adaptive_noise_fn": "rule_of_thumb_0.50"
                    },
                },
            )
            for simulator in simulators
        ]
    )
    fig = gof_result.plot_metric(
        plot_dict, metric="score", figsize=(12, 4.5), layout=(1, 3), log_scale_y=False
    )
    plt.suptitle('noise schedules %s'%estimator)
    fig.savefig(os.path.join(os.path.join(DATA_DIR_LOCAL, EXP_PREFIX), "noise_schedules_rule_of_thumb_%s.png"%estimator))

#
for estimator in estimators:
    plot_dict = dict(
        [
            (
                simulator,
                {
                    "fixed_rate_0.40": {
                        "simulator": simulator,
                        "estimator": estimator,
                        "adaptive_noise_fn": "fixed_rate_0.40"
                    },
                    "fixed_rate_0.20": {
                        "simulator": simulator,
                        "estimator": estimator,
                        "adaptive_noise_fn": "fixed_rate_0.20"
                    },
                    "fixed_rate_0.10": {
                        "simulator": simulator,
                        "estimator": estimator,
                        "adaptive_noise_fn": "fixed_rate_0.10"
                    },
                },
            )
            for simulator in simulators
        ]
    )
    fig = gof_result.plot_metric(
        plot_dict, metric="score", figsize=(12, 4.5), layout=(1, 3), log_scale_y=False
    )
    plt.suptitle('noise schedules %s'%estimator)
    fig.savefig(os.path.join(os.path.join(DATA_DIR_LOCAL, EXP_PREFIX), "noise_schedules_fixed_rate_%s.png"%estimator))


for estimator in estimators:
    plot_dict = dict(
        [
            (
                simulator,
                {
                    "quadratic_rate_5.0": {
                        "simulator": simulator,
                        "estimator": estimator,
                        "adaptive_noise_fn": "quadratic_rate_5.00"
                    },
                    "quadratic_rate_2.0": {
                        "simulator": simulator,
                        "estimator": estimator,
                        "adaptive_noise_fn": "quadratic_rate_2.00"
                    },
                    "quadratic_rate_1.0": {
                        "simulator": simulator,
                        "estimator": estimator,
                        "adaptive_noise_fn": "quadratic_rate_1.00"
                    },
                },
            )
            for simulator in simulators
        ]
    )
    fig = gof_result.plot_metric(
        plot_dict, metric="score", figsize=(12, 4.5), layout=(1, 3), log_scale_y=False
    )
    plt.suptitle('noise schedules %s'%estimator)
    fig.savefig(os.path.join(os.path.join(DATA_DIR_LOCAL, EXP_PREFIX), "noise_schedules_quadratic_rate_%s.png"%estimator))

for estimator in estimators:
    plot_dict = dict(
        [
            (
                simulator,
                {
                    "rule_of_thumb_5.0": {
                        "simulator": simulator,
                        "estimator": estimator,
                        "adaptive_noise_fn": "rule_of_thumb_0.70"
                    },
                    "quadratic_rate_5.0": {
                        "simulator": simulator,
                        "estimator": estimator,
                        "adaptive_noise_fn": "quadratic_rate_5.00"
                    },
                    "fixed_rate_0.2": {
                        "simulator": simulator,
                        "estimator": estimator,
                        "adaptive_noise_fn": "fixed_rate_0.20"
                    },
                },
            )
            for simulator in simulators
        ]
    )
    fig = gof_result.plot_metric(
        plot_dict, metric="score", figsize=(12, 4.5), layout=(1, 3), log_scale_y=False
    )
    fig.axes[0].set_ylim((-2.2, -1.93))
    fig.axes[2].set_ylim((1.3, 1.63))
    plt.suptitle('noise schedules %s'%estimator)
    fig.savefig(os.path.join(os.path.join(DATA_DIR_LOCAL, EXP_PREFIX), "noise_schedules_comparison_%s.png"%estimator))


