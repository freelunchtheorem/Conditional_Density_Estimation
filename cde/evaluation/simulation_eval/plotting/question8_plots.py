from ml_logger import logger

from cde.model_fitting.GoodnessOfFitResults import GoodnessOfFitResults
from cde.evaluation.simulation_eval import base_experiment
import cde.model_fitting.ConfigRunner as ConfigRunner
import matplotlib.pyplot as plt
import os

EXP_PREFIX = "question8_benchmark"
RESULTS_FILE = "results.pkl"
CLUSTER_DIR = "/local/rojonas/cde/data/local"
LOCATION = "{}/{}/{}".format(CLUSTER_DIR, EXP_PREFIX, RESULTS_FILE)
DATA_DIR_LOCAL = "/home/jonasrothfuss/Dropbox/Eigene_Dateien/ETH/02_Projects/02_Noise_Regularization/02_Code_Conditional_Density_Estimation/data/cluster"

logger.configure(
    #"/local/rojonas/cde/data/local",
    DATA_DIR_LOCAL,
    #CLUSTER_DIR,
    EXP_PREFIX,
)

results_from_pkl_file = dict(logger.load_pkl_log(RESULTS_FILE))
gof_result = GoodnessOfFitResults(single_results_dict=results_from_pkl_file)
results_df = gof_result.generate_results_dataframe(base_experiment.KEYS_OF_INTEREST_LOGPROB + ["bandwidth", "param_selection"])
results_df.replace(to_replace=[None], value="None", inplace=True)

# seprarate 2d and 4d GMM
results_df.index = list(range(len(results_df)))

for i, row in results_df[['simulator', 'ndim_y']].iterrows():
    if row['simulator'] == 'GaussianMixture':
        results_df.at[i, 'simulator'] = '%s_%id'%(row['simulator'], row['ndim_y'])

estimators = [
    "MixtureDensityNetwork",
    "KernelMixtureNetwork",
    "NormalizingFlowEstimator",
    "ConditionalKernelDensityEstimation",
    "NeighborKernelDensityEstimation",
    "LSConditionalDensityEstimation"
]
simulators = ["GaussianMixture_2d", "SkewNormal"]

plot_dict = dict(
    [
        (
            simulator,
            {
                "MDN": {
                    "simulator": simulator,
                    "estimator": "MixtureDensityNetwork",
                    "adaptive_noise_fn": "polynomial_rate_3_1.00" if simulator == "SkewNormal" else
                                         ("polynomial_rate_2_1.00" if simulator=="GaussianMixture_2d" else "polynomial_rate_2_2.00")
                },
                "KMN": {
                    "simulator": simulator,
                    "estimator": "KernelMixtureNetwork",
                    "adaptive_noise_fn": "polynomial_rate_2_2.00" if simulator == "SkewNormal" else
                        ("polynomial_rate_1_1.00" if simulator == "GaussianMixture_2d" else "polynomial_rate_2_2.00")
                },
                "NF": {
                    "simulator": simulator,
                    "estimator": "NormalizingFlowEstimator",
                    "adaptive_noise_fn": "polynomial_rate_2_1.00" if simulator == "SkewNormal" else
                        ("polynomial_rate_1_1.00" if simulator == "GaussianMixture_2d" else "polynomial_rate_2_2.00")
                },
                "CKDE": {
                    "simulator": simulator,
                    "estimator": "ConditionalKernelDensityEstimation",
                    "bandwidth": "normal_reference"
                },
                "NKDE": {
                    "simulator": simulator,
                    "estimator": "NeighborKernelDensityEstimation",
                    "param_selection": "normal_reference"
                },
                "LSCDE": {
                    "simulator": simulator,
                    "estimator": "LSConditionalDensityEstimation",
                },
            },
        )
        for simulator in simulators
    ]
)
FIGSIZE = (10, 4.5)

fig = gof_result.plot_metric(
    plot_dict, metric="score", figsize=FIGSIZE, layout=(1, 2), log_scale_y=False
)
fig.tight_layout()

for i in range(len(fig.axes)):
    fig.axes[i].set_title('')
    fig.axes[i].set_xlabel('')
    fig.axes[i].set_ylabel('')

fig.axes[0].set_xlabel('number of train observations')
fig.axes[0].set_ylabel('log-likelihood')

fig.axes[0].get_legend().remove()
fig.axes[1].set_ylim((0.9, 1.62))

fig.axes[0].set_title('Gaussian Mixture')
fig.axes[1].set_title('Skew Normal')

fig.savefig(os.path.join(os.path.join(DATA_DIR_LOCAL, EXP_PREFIX), "cde_benchmarks.png"))
fig.savefig(os.path.join(os.path.join(DATA_DIR_LOCAL, EXP_PREFIX), "cde_benchmarks.pdf"))




