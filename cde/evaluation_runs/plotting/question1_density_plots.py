from ml_logger import logger
from cde.evaluation.GoodnessOfFitResults import GoodnessOfFitResults
from cde.helpers import take_of_type
import cde.evaluation.ConfigRunner as ConfigRunner
from cde.evaluation.plotting import get_density_plots
import tensorflow as tf
import matplotlib.pyplot as plt





if __name__ == '__main__':
  prob_model_params_to_visualize = {'heteroscedastic': True, 'random_seed': 20, 'std': 1, 'simulator': 'EconDensity'}
  estimators_to_visualize = ['KernelMixtureNetwork', 'MixtureDensityNetwork']
  path_to_results="/Volumes/fabioexternal/Dropbox/0_Studium/Master/git_projects/Nonparametric_Density_Estimation/data/cluster"

  get_density_plots(estimators_list=estimators_to_visualize, simulators_dict=prob_model_params_to_visualize,
                    path_to_results=path_to_results, task_ids=[0, 16])