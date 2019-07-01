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


# comparison compact

plot_dict = dict(
    [
        (
            "GaussianMixture_%s" % estimator,
            {
                "rule_of_thumb": {
                    "simulator": "GaussianMixture",
                    "estimator": estimator,
                    "adaptive_noise_fn": "rule_of_thumb_0.50"
                },
                "quadratic_rate": {
                    "simulator": "GaussianMixture",
                    "estimator": estimator,
                    "adaptive_noise_fn": "quadratic_rate_1.00"
                },
                "fixed_rate": {
                    "simulator": "GaussianMixture",
                    "estimator": estimator,
                    "adaptive_noise_fn": "fixed_rate_0.20"
                },
                "no noise": {
                    "simulator": "GaussianMixture",
                    "estimator": estimator,
                    "adaptive_noise_fn": "fixed_rate_0.00"
                },
            },
        ) for estimator in estimators] + [
        (
            "SkewNormal_%s" % estimator,
            {
                "rule_of_thumb": {
                    "simulator": "SkewNormal",
                    "estimator": estimator,
                    "adaptive_noise_fn": "rule_of_thumb_0.50" if estimator == "NormalizingFlowEstimator" else "rule_of_thumb_0.70"
                },
                "quadratic_rate": {
                    "simulator": "SkewNormal",
                    "estimator": estimator,
                    "adaptive_noise_fn": "quadratic_rate_5.00"
                },
                "fixed_rate": {
                    "simulator": "SkewNormal",
                    "estimator": estimator,
                    "adaptive_noise_fn": "fixed_rate_0.20"
                },
                "no noise": {
                    "simulator": "SkewNormal",
                    "estimator": estimator,
                    "adaptive_noise_fn": "fixed_rate_0.00"
                },
            },
        ) for estimator in estimators]
)
fig = gof_result.plot_metric(
    plot_dict, metric="score", figsize=(12, 6), layout=(2, 3), log_scale_y=False
)

for i in range(6):
    fig.axes[i].set_title("")
    fig.axes[i].set_xlabel('')
    fig.axes[i].set_ylabel('')
    fig.axes[i].get_legend().remove()

fig.axes[0].set_title("Mixture Density Network")
fig.axes[1].set_title("Kernel Mixture Network")
fig.axes[2].set_title("Normalizing Flow")

fig.axes[0].set_xlabel('number of train observations')
fig.axes[0].set_ylabel('log-likelihood')

for i in range(0, 3):
    fig.axes[i].set_ylim((-4.6, -2.9))

for i in range(3, 6):
    fig.axes[i].set_ylim((1.34, 1.63))

fig.axes[0].annotate("Gaussian Mixture", xy=(0, 0.5), xytext=(-fig.axes[0].yaxis.labelpad - 6.0, 0),
                     xycoords=fig.axes[0].yaxis.label, textcoords='offset points',
                     size='large', ha='right', va='center', rotation=90)
fig.axes[3].annotate("Skew Normal", xy=(0, 0.5), xytext=(-fig.axes[0].yaxis.labelpad - 22.0, 0),
                     xycoords=fig.axes[3].yaxis.label, textcoords='offset points',
                     size='large', ha='right', va='center', rotation=90)

plt.legend(["rule of thumb", "sqrt rate", "fixed rate", "no noise"], loc='lower right')
fig.tight_layout()

fig.savefig(os.path.join(os.path.join(DATA_DIR_LOCAL, EXP_PREFIX), "noise_schedules_comparison.png"))
fig.savefig(os.path.join(os.path.join(DATA_DIR_LOCAL, EXP_PREFIX), "noise_schedules_comparison.pdf"))



# # rules of thumb
# for estimator in estimators:
#     plot_dict = dict(
#         [
#             (
#                 simulator,
#                 {
#                     "rule_of_thumb_1.0": {
#                         "simulator": simulator,
#                         "estimator": estimator,
#                         "adaptive_noise_fn": "rule_of_thumb_1.00"
#                     },
#                     "rule_of_thumb_0.7": {
#                         "simulator": simulator,
#                         "estimator": estimator,
#                         "adaptive_noise_fn": "rule_of_thumb_0.70"
#                     },
#                     "rule_of_thumb_0.5": {
#                         "simulator": simulator,
#                         "estimator": estimator,
#                         "adaptive_noise_fn": "rule_of_thumb_0.50"
#                     },
#                 },
#             )
#             for simulator in simulators
#         ]
#     )
#     fig = gof_result.plot_metric(
#         plot_dict, metric="score", figsize=(12, 4.5), layout=(1, 3), log_scale_y=False
#     )
#     plt.suptitle('noise schedules %s'%estimator)
#     fig.savefig(os.path.join(os.path.join(DATA_DIR_LOCAL, EXP_PREFIX), "noise_schedules_rule_of_thumb_%s.png"%estimator))
#
#
# # fixed noise intensity
#
# for estimator in estimators:
#     plot_dict = dict(
#         [
#             (
#                 simulator,
#                 {
#                     "fixed_rate_0.40": {
#                         "simulator": simulator,
#                         "estimator": estimator,
#                         "adaptive_noise_fn": "fixed_rate_0.40"
#                     },
#                     "fixed_rate_0.20": {
#                         "simulator": simulator,
#                         "estimator": estimator,
#                         "adaptive_noise_fn": "fixed_rate_0.20"
#                     },
#                     "fixed_rate_0.10": {
#                         "simulator": simulator,
#                         "estimator": estimator,
#                         "adaptive_noise_fn": "fixed_rate_0.10"
#                     },
#                 },
#             )
#             for simulator in simulators
#         ]
#     )
#     fig = gof_result.plot_metric(
#         plot_dict, metric="score", figsize=(12, 4.5), layout=(1, 3), log_scale_y=False
#     )
#     plt.suptitle('noise schedules %s'%estimator)
#     fig.savefig(os.path.join(os.path.join(DATA_DIR_LOCAL, EXP_PREFIX), "noise_schedules_fixed_rate_%s.png"%estimator))
#
# # quadratic rate
#
# for estimator in estimators:
#     plot_dict = dict(
#         [
#             (
#                 simulator,
#                 {
#                     "quadratic_rate_5.0": {
#                         "simulator": simulator,
#                         "estimator": estimator,
#                         "adaptive_noise_fn": "quadratic_rate_5.00"
#                     },
#                     "quadratic_rate_2.0": {
#                         "simulator": simulator,
#                         "estimator": estimator,
#                         "adaptive_noise_fn": "quadratic_rate_2.00"
#                     },
#                     "quadratic_rate_1.0": {
#                         "simulator": simulator,
#                         "estimator": estimator,
#                         "adaptive_noise_fn": "quadratic_rate_1.00"
#                     },
#                 },
#             )
#             for simulator in simulators
#         ]
#     )
#     fig = gof_result.plot_metric(
#         plot_dict, metric="score", figsize=(12, 4.5), layout=(1, 3), log_scale_y=False
#     )
#     plt.suptitle('noise schedules %s'%estimator)
#     fig.savefig(os.path.join(os.path.join(DATA_DIR_LOCAL, EXP_PREFIX), "noise_schedules_quadratic_rate_%s.png"%estimator))
#
#
# # polynomial rate
#
# for estimator in estimators:
#     plot_dict = dict(
#         [
#             (
#                 simulator,
#                 {
#                     "polynomial_rate_2_1.00": {
#                         "simulator": simulator,
#                         "estimator": estimator,
#                         "adaptive_noise_fn": "polynomial_rate_2_1.00"
#                     },
#                     "polynomial_rate_2_2.00": {
#                         "simulator": simulator,
#                         "estimator": estimator,
#                         "adaptive_noise_fn": "polynomial_rate_2_2.00"
#                     },
#                     "polynomial_rate_3_1.00": {
#                         "simulator": simulator,
#                         "estimator": estimator,
#                         "adaptive_noise_fn": "polynomial_rate_3_1.00"
#                     },
#                     "polynomial_rate_3_2.00": {
#                         "simulator": simulator,
#                         "estimator": estimator,
#                         "adaptive_noise_fn": "polynomial_rate_3_2.00"
#                     },
#                 },
#             )
#             for simulator in simulators
#         ]
#     )
#     fig = gof_result.plot_metric(
#         plot_dict, metric="score", figsize=(12, 4.5), layout=(1, 3), log_scale_y=False
#     )
#     fig.axes[0].set_ylim((-2.2, -1.93))
#     fig.axes[2].set_ylim((1.3, 1.63))
#     plt.suptitle('noise schedules %s'%estimator)
#     fig.savefig(os.path.join(os.path.join(DATA_DIR_LOCAL, EXP_PREFIX), "noise_schedules_polynomial_%s.png"%estimator))

# comparison

# for estimator in estimators:
#     plot_dict = dict(
#         [
#             ("EconDensity",
#                 {
#                 "rule_of_thumb": {
#                     "simulator": "EconDensity",
#                     "estimator": estimator,
#                     "adaptive_noise_fn": "rule_of_thumb_0.50" if estimator == "NormalizingFlowEstimator" else "rule_of_thumb_1.00"
#                 },
#                 "quadratic_rate": {
#                     "simulator": "EconDensity",
#                     "estimator": estimator,
#                     "adaptive_noise_fn": "quadratic_rate_5.00"
#                 },
#                 "fixed_rate": {
#                     "simulator": "EconDensity",
#                     "estimator": estimator,
#                     "adaptive_noise_fn": "fixed_rate_0.20"
#                 },
#                 "poly_rate": {
#                     "simulator": "EconDensity",
#                     "estimator": estimator,
#                     "adaptive_noise_fn": "polynomial_rate_2_2.00"
#                 },
#                 },
#             ),
#
#             (
#                 "GaussianMixture",
#                     {
#                     "rule_of_thumb": {
#                         "simulator": "GaussianMixture",
#                         "estimator": estimator,
#                         "adaptive_noise_fn": "rule_of_thumb_0.50"
#                     },
#                     "quadratic_rate": {
#                         "simulator": "GaussianMixture",
#                         "estimator": estimator,
#                         "adaptive_noise_fn": "quadratic_rate_1.00"
#                     },
#                     "fixed_rate": {
#                         "simulator": "GaussianMixture",
#                         "estimator": estimator,
#                         "adaptive_noise_fn": "fixed_rate_0.20"
#                     },
#                     "poly_rate": {
#                         "simulator": "GaussianMixture",
#                         "estimator": estimator,
#                         "adaptive_noise_fn": "polynomial_rate_2_1.00"
#                     },
#                 },
#             ),
#
#                     (
#                     "SkewNormal",
#                     {
#                     "rule_of_thumb": {
#                         "simulator": "SkewNormal",
#                         "estimator": estimator,
#                         "adaptive_noise_fn": "rule_of_thumb_0.50" if estimator == "NormalizingFlowEstimator" else "rule_of_thumb_0.70"
#                     },
#                     "quadratic_rate": {
#                         "simulator": "SkewNormal",
#                         "estimator": estimator,
#                         "adaptive_noise_fn": "quadratic_rate_5.00"
#                     },
#                     "fixed_rate": {
#                         "simulator": "SkewNormal",
#                         "estimator": estimator,
#                         "adaptive_noise_fn": "fixed_rate_0.20"
#                     },
#                     "poly_rate": {
#                         "simulator": "SkewNormal",
#                         "estimator": estimator,
#                         "adaptive_noise_fn": "polynomial_rate_2_2.00" if estimator == "KernelMixtureNetwork" else
#                         (
#                             "polynomial_rate_3_1.00" if estimator == "MixtureDensityNetwork" else "polynomial_rate_2_1.00")
#                     },
#                     },
#                 ),
#         ]
#     )
#     fig = gof_result.plot_metric(
#         plot_dict, metric="score", figsize=(12, 4.5), layout=(1, 3), log_scale_y=False
#     )
#     fig.axes[0].set_ylim((-2.2, -1.93))
#     fig.axes[2].set_ylim((1.3, 1.63))
#     plt.suptitle('noise schedules %s'%estimator)
#     fig.savefig(os.path.join(os.path.join(DATA_DIR_LOCAL, EXP_PREFIX), "noise_schedules_comparison_%s.png"%estimator))
#     fig.savefig(os.path.join(os.path.join(DATA_DIR_LOCAL, EXP_PREFIX), "noise_schedules_comparison_%s.pdf" % estimator))