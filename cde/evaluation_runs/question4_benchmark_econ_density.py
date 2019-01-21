import matplotlib as mpl
mpl.use("PS") #handles X11 server detection (required to run on console)
import numpy as np

from cde.evaluation.GoodnessOfFitResults import GoodnessOfFitResults
from cde.evaluation_runs import base_experiment
import cde.evaluation.ConfigRunner as ConfigRunner
import config

from ml_logger import logger

EXP_PREFIX = 'question4_benchmark_econ_density'
RESULTS_FILE = 'results.pkl'

N_MC_SAMPLES = int(10**5)


def question4():
  estimator_params = {
    'ConditionalKernelDensityEstimation':
      {
          'bandwidth': ['normal_reference', 'cv_ml'],
      },
    'NeighborKernelDensityEstimation':
      {},
    'LSConditionalDensityEstimation':
      {},
    'MixtureDensityNetwork':
      {
        'n_centers': [20],
        'n_training_epochs': [2000],
        'hidden_sizes': [(16, 16)],
        'x_noise_std': [0.1],
        'y_noise_std': [0.1],
        'random_seed': [22]
      },
    'KernelMixtureNetwork':
      {'center_sampling_method': ["k_means"],
       'n_centers': [50],
       'keep_edges': [True],
       'init_scales': [[0.3, 0.7]],
       'train_scales': [True],
       'hidden_sizes': [(16, 16)],
       'n_training_epochs': [1000],
       'x_noise_std': [0.1],
       'y_noise_std': [0.1],
       },
  }

  simulators_params = {
  'EconDensity': {'std': [1],
                  'heteroscedastic': [True]
                  },
  }

  observations = 100 * np.logspace(2, 6, num=7, base=2.0, dtype=np.int32)

  return estimator_params, simulators_params, observations


if __name__ == '__main__':
  estimator_params, simulators_params, observations = question4()
  load = base_experiment.launch_experiment(estimator_params, simulators_params, observations, EXP_PREFIX,
                                           n_mc_samples=N_MC_SAMPLES, tail_measures=True)

  if load:
    logger.configure(config.DATA_DIR, EXP_PREFIX)

    results_from_pkl_file = dict(logger.load_pkl_log(RESULTS_FILE))
    gof_result = GoodnessOfFitResults(single_results_dict=results_from_pkl_file)
    results_df = gof_result.generate_results_dataframe(base_experiment.KEYS_OF_INTEREST)

    gof_result = ConfigRunner.load_dumped_estimators(gof_result)