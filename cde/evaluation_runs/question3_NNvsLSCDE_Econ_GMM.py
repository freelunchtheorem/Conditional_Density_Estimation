import matplotlib as mpl
mpl.use("PS") #handles X11 server detection (required to run on console)
import numpy as np

from cde.evaluation.GoodnessOfFitResults import GoodnessOfFitResults
from cde.evaluation_runs import base_experiment

from ml_logger import logger

EXP_PREFIX = 'question3_NNvsLSCDE_Econ_GMM'
RESULTS_FILE = 'results.pkl'




def question3():
    # todo: add KMN & MDN

    estimator_params = {
    'LSConditionalDensityEstimation':
      {
          'center_sampling_method': ["all", "distance", "k_means", "agglomerative", "random"],
          'bandwidth': [0.3, 0.5, 0.7],
          'regularization': [0.1, 0.2, 0.5, 1.0],
          'n_centers': [100, 300, 600, 1000],
          'random_seed': [22]
      },
    }

    simulators_params = {
        'EconDensity': {
            'std': [1],
            'heteroscedastic': [True]
        },

        'GaussianMixture': {
            'n_kernels' : [10],
            'ndim_x': [2],
            'ndim_y': [2],
            'means_std': [1.5]
        }
    }

    observations = 100 * np.logspace(0, 6, num=7, base=2.0, dtype=np.int32)

    return estimator_params, simulators_params, observations


if __name__ == '__main__':
    estimator_params, simulators_params, observations = question3()
    load = base_experiment.launch_experiment(estimator_params, simulators_params, observations, EXP_PREFIX)

    if load:
        results_from_pkl_file = dict(logger.load_pkl_log(RESULTS_FILE))
        gof_result = GoodnessOfFitResults(single_results_dict=results_from_pkl_file)
        results_df = gof_result.generate_results_dataframe(base_experiment.KEYS_OF_INTEREST)

        graph_dicts = [
            {"estimator": "LSConditionalDensityEstimation"},
            {"estimator": "LSConditionalDensityEstimation"}
            ]

        gof_result.plot_metric(graph_dicts, metric="js_divergence")
        print(results_df)

