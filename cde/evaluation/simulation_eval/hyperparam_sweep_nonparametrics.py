import matplotlib as mpl
import numpy as np
mpl.use("PS") #handles X11 server detection (required to run on console)

from cde.evaluation.simulation_eval import base_experiment

EXP_PREFIX = 'hyperparam_sweep_nonparam'
N_MC_SAMPLES = int(10**5)

def question1():
  estimator_params = {
    'NeighborKernelDensityEstimation':
      {'bandwidth': [0.1, 0.4, 0.7, 1.0],
        'param_selection': [None, 'normal_reference', 'cv_ml'],
       'epsilon': [0.05, 0.2, 0.4, 0.6]},
    'LSConditionalDensityEstimation':
      {'bandwidth': [0.2, 0.5, 0.7, 1.0],
      'regularization': [0.1, 0.5, 1.0, 4.0, 8.0]},
  }

  simulators_params = {
  'EconDensity': {'std': [1],
                  'heteroscedastic': [True],
                  },
  'ArmaJump': {'c': [0.1],
               'arma_a1': [0.9],
               'std': [0.05],
               'jump_prob': [0.05],
              },
  'SkewNormal': {}
  }

  observations = 100 * np.logspace(3, 6, num=4, base=2.0, dtype=np.int32)

  return estimator_params, simulators_params, observations


if __name__ == '__main__':
    estimator_params, simulators_params, observations = question1()
    base_experiment.launch_experiment(estimator_params, simulators_params, observations, EXP_PREFIX,
                                      n_mc_samples=N_MC_SAMPLES, tail_measures=False)


