import matplotlib as mpl
mpl.use("PS") #handles X11 server detection (required to run on console)
import numpy as np

from cde.evaluation.GoodnessOfFitResults import GoodnessOfFitResults
from cde.evaluation_runs import base_experiment

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
