import matplotlib as mpl
mpl.use("PS") #handles X11 server detection (required to run on console)
import numpy as np

from cde.evaluation.GoodnessOfFitResults import GoodnessOfFitResults
from cde.evaluation_runs import base_experiment
import cde.evaluation.ConfigRunner as ConfigRunner

from ml_logger import logger

EXP_PREFIX = 'question3_KDE'
RESULTS_FILE = 'results.pkl'




def question3():
  estimator_params = {

    'ConditionalKernelDensityEstimation':
      {
          'bandwidth_selection': ['normal_reference', 'cv_ml'],
          'random_seed': [22]
      },
  }

  simulators_params = {
  'EconDensity': {'std': [1],
                  'heteroscedastic': [True]
                  },

  'ArmaJump': {'c': [0.1],
               'arma_a1': [0.9],
               'std': [0.05],
               'jump_prob': [0.05]
               },

  'GaussianMixture': {'n_kernels': [10],
                      'ndim_x': [2],
                      'ndim_y': [2],
                      'means_std': [1.5]
                },
  'SkewNormal': {}
  }

  observations = 100 * np.logspace(0, 6, num=7, base=2.0, dtype=np.int32)

  return estimator_params, simulators_params, observations


if __name__ == '__main__':
  estimator_params, simulators_params, observations = question3()
  load = base_experiment.launch_experiment(estimator_params, simulators_params, observations, EXP_PREFIX)

  if load:
    logger.configure('/Users/fabioferreira/Dropbox/0_Studium/Master/git_projects/Nonparametric_Density_Estimation/data/local', EXP_PREFIX)

    results_from_pkl_file = dict(logger.load_pkl_log(RESULTS_FILE))
    gof_result = GoodnessOfFitResults(single_results_dict=results_from_pkl_file)
    results_df = gof_result.generate_results_dataframe(base_experiment.KEYS_OF_INTEREST)

    gof_result = ConfigRunner.load_dumped_estimators(gof_result)