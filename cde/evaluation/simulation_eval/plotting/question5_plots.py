from ml_logger import logger

from cde.model_fitting.GoodnessOfFitResults import GoodnessOfFitResults
from cde.evaluation.simulation_eval import base_experiment
import cde.model_fitting.ConfigRunner as ConfigRunner
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import os

EXP_PREFIX = "question5_benchmark"
RESULTS_FILE = "results.pkl"
CLUSTER_DIR = "/home/simon/Documents/KIT/Informatik/Bachelorarbeit/Conditional_Density_Estimation/data/cluster"
LOCATION = "{}/{}/{}".format(CLUSTER_DIR, EXP_PREFIX, RESULTS_FILE)

logger.configure(
    "/home/simon/Documents/KIT/Informatik/Bachelorarbeit/Conditional_Density_Estimation/data/cluster",
    EXP_PREFIX,
)

with open(LOCATION, "rb") as fh:
    results_from_pkl_file = pickle.load(fh)
gof_result = GoodnessOfFitResults(single_results_dict=results_from_pkl_file)
results_df = gof_result.generate_results_dataframe(base_experiment.KEYS_OF_INTEREST)
results_df.replace(to_replace=[None], value="None", inplace=True)

estimators = [
    "MixtureDensityNetwork",
    "KernelMixtureNetwork",
    "NormalizingFlowEstimator",
]
simulators = ["ArmaJump", "EconDensity", "GaussianMixture", "SkewNormal"]


plot_dict = dict(
    [
        (
            simulator,
            {
                # "MDN": {"simulator": simulator, "estimator": "MixtureDensityNetwork"},
                "NF_no_reg": {
                    "simulator": simulator,
                    "estimator": "NormalizingFlowEstimator",
                    "weight_normalization": False,
                    "dropout": 0.0,
                    "weight_decay": 0.0,
                    "x_noise_std": "None",
                    "y_noise_std": "None",
                },
                "NF_weight_decay": {
                    "simulator": simulator,
                    "estimator": "NormalizingFlowEstimator",
                    "weight_normalization": False,
                    "dropout": 0.0,
                    "weight_decay": 5e-5,
                    "x_noise_std": "None",
                    "y_noise_std": "None",
                },
                "NF_dropout": {
                    "simulator": simulator,
                    "estimator": "NormalizingFlowEstimator",
                    "weight_normalization": True,
                    "dropout": 0.2,
                    "weight_decay": 0.0,
                    "x_noise_std": "None",
                    "y_noise_std": "None",
                },
                "NF_noise": {
                    "simulator": simulator,
                    "estimator": "NormalizingFlowEstimator",
                    "weight_normalization": True,
                    "dropout": 0.0,
                    "weight_decay": 0.0,
                    "x_noise_std": 0.1,
                    "y_noise_std": 0.1,
                },
                "NF_all": {
                    "simulator": simulator,
                    "estimator": "NormalizingFlowEstimator",
                    "weight_normalization": False,
                    "dropout": 0.0,
                    "weight_decay": 5e-5,
                    "x_noise_std": 0.1,
                    "y_noise_std": 0.1,
                },
            },
        )
        for simulator in simulators
    ]
)
fig = gof_result.plot_metric(
    plot_dict, metric="hellinger_distance", figsize=(14, 10), layout=(2, 2)
)
fig.savefig(os.path.join(EXP_PREFIX, "nf_compare.pdf"))
fig.savefig(os.path.join(EXP_PREFIX, "nf_compare.png"))


plot_dict = dict(
    [
        (
            simulator,
            {
                "MDN_baseline": {
                    "simulator": simulator,
                    "estimator": "MixtureDensityNetwork",
                    "weight_normalization": True,
                    "dropout": 0.0,
                    "weight_decay": 0.0,
                    "x_noise_std": 0.2,
                    "y_noise_std": 0.1,
                },
                "MDN_no_reg": {
                    "simulator": simulator,
                    "estimator": "MixtureDensityNetwork",
                    "weight_normalization": False,
                    "dropout": 0.0,
                    "weight_decay": 0.0,
                    "x_noise_std": "None",
                    "y_noise_std": "None",
                },
                "MDN_weight_decay": {
                    "simulator": simulator,
                    "estimator": "MixtureDensityNetwork",
                    "weight_normalization": False,
                    "dropout": 0.0,
                    "weight_decay": 5e-5,
                    "x_noise_std": "None",
                    "y_noise_std": "None",
                },
                "MDN_dropout": {
                    "simulator": simulator,
                    "estimator": "MixtureDensityNetwork",
                    "weight_normalization": True,
                    "dropout": 0.2,
                    "weight_decay": 0.0,
                    "x_noise_std": "None",
                    "y_noise_std": "None",
                },
                "MDN_all": {
                    "simulator": simulator,
                    "estimator": "MixtureDensityNetwork",
                    "weight_normalization": True,
                    "dropout": 0.2,
                    "weight_decay": 0.0,
                    "x_noise_std": 0.1,
                    "y_noise_std": 0.1,
                },
            },
        )
        for simulator in simulators
    ]
)
fig = gof_result.plot_metric(
    plot_dict, metric="hellinger_distance", figsize=(14, 10), layout=(2, 2)
)
fig.savefig(os.path.join(EXP_PREFIX, "mdn_compare.pdf"))
fig.savefig(os.path.join(EXP_PREFIX, "mdn_compare.png"))


plot_dict = dict(
    [
        (
            simulator,
            {
                # "MDN": {"simulator": simulator, "estimator": "MixtureDensityNetwork"},
                "KMN_baseline": {
                    "simulator": simulator,
                    "estimator": "KernelMixtureNetwork",
                    "weight_normalization": True,
                    "dropout": 0.0,
                    "weight_decay": 0.0,
                    "x_noise_std": 0.2,
                    "y_noise_std": 0.1,
                },
                "KMN_weight_norm": {
                    "simulator": simulator,
                    "estimator": "KernelMixtureNetwork",
                    "weight_normalization": True,
                    "dropout": 0.0,
                    "weight_decay": 0.0,
                    "x_noise_std": "None",
                    "y_noise_std": "None",
                },
                "KMN_weight_decay": {
                    "simulator": simulator,
                    "estimator": "KernelMixtureNetwork",
                    "weight_normalization": False,
                    "dropout": 0.0,
                    "weight_decay": 5e-5,
                    "x_noise_std": "None",
                    "y_noise_std": "None",
                },
                "KMN_dropout": {
                    "simulator": simulator,
                    "estimator": "KernelMixtureNetwork",
                    "weight_normalization": True,
                    "dropout": 0.2,
                    "weight_decay": 0.0,
                    "x_noise_std": "None",
                    "y_noise_std": "None",
                },
                "KMN_all": {
                    "simulator": simulator,
                    "estimator": "KernelMixtureNetwork",
                    "weight_normalization": False,
                    "dropout": 0.2,
                    "weight_decay": 5e-5,
                    "x_noise_std": 0.1,
                    "y_noise_std": 0.01,
                },
            },
        )
        for simulator in simulators
    ]
)
fig = gof_result.plot_metric(
    plot_dict, metric="hellinger_distance", figsize=(14, 10), layout=(2, 2)
)
fig.savefig(os.path.join(EXP_PREFIX, "kmn_compare.pdf"))
fig.savefig(os.path.join(EXP_PREFIX, "kmn_compare.png"))


plot_dict = dict(
    [
        (
            simulator,
            {
                "NF_all": {
                    "simulator": simulator,
                    "estimator": "NormalizingFlowEstimator",
                    "weight_normalization": False,
                    "dropout": 0.0,
                    "weight_decay": 5e-5,
                    "x_noise_std": 0.1,
                    "y_noise_std": 0.1,
                },
                "MDN_all": {
                    "simulator": simulator,
                    "estimator": "MixtureDensityNetwork",
                    "weight_normalization": True,
                    "dropout": 0.2,
                    "weight_decay": 0.0,
                    "x_noise_std": 0.1,
                    "y_noise_std": 0.1,
                },
                "KMN_all": {
                    "simulator": simulator,
                    "estimator": "KernelMixtureNetwork",
                    "weight_normalization": False,
                    "dropout": 0.2,
                    "weight_decay": 5e-5,
                    "x_noise_std": 0.1,
                    "y_noise_std": 0.01,
                },
            },
        )
        for simulator in simulators
    ]
)
fig = gof_result.plot_metric(
    plot_dict, metric="hellinger_distance", figsize=(14, 10), layout=(2, 2)
)
fig.savefig(os.path.join(EXP_PREFIX, "all_compare.pdf"))
fig.savefig(os.path.join(EXP_PREFIX, "all_compare.png"))


graph_dict = {
    "simulator": "ArmaJump",
    "estimator": "NormalizingFlowEstimator",
    "weight_normalization": False,
    "dropout": 0.0,
    "weight_decay": 5e-5,
    "x_noise_std": 0.1,
    "y_noise_std": 0.01,
    "n_observations": 1600,
}

sub_df = results_df.loc[
    (results_df[list(graph_dict)] == pd.Series(graph_dict)).all(axis=1)
]


for simulator in simulators:
    for estimator in ["MixtureDensityNetwork", "NormalizingFlowEstimator"]:
        print("\n\nESTIMATOR: {}  SIMULATOR: {} \n".format(estimator, simulator))
        da = (
            results_df.loc[
                (results_df["estimator"] == estimator)
                & (results_df["simulator"] == simulator)
                & (results_df["n_observations"] == 1600)
                & (results_df["weight_normalization"] == False)
            ]
            .groupby(["x_noise_std", "y_noise_std", "dropout", "weight_decay"])[
                "hellinger_distance"
            ]
            .mean()
            .sort_values()[0:5]
        )
        print(da)


for estimator in estimators:
    for simulator in simulators:
        print("\n\nESTIMATOR: {}  SIMULATOR: {} \n".format(estimator, simulator))
        da = (
            results_df.loc[
                (results_df["estimator"] == estimator)
                & (results_df["simulator"] == simulator)
                & (results_df["n_observations"] == 1600)
                & (results_df["weight_normalization"] == True)
            ]
            .groupby(["x_noise_std", "y_noise_std", "dropout", "weight_decay"])[
                "hellinger_distance"
            ]
            .mean()
            .sort_values()[0:5]
        )
        print(da)
