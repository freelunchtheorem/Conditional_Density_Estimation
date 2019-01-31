from ml_logger import logger

from cde.model_fitting.GoodnessOfFitResults import GoodnessOfFitResults
from cde.evaluation.simulation_eval import base_experiment
import cde.model_fitting.ConfigRunner as ConfigRunner
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXP_PREFIX = 'question2_entropy_reg'
RESULTS_FILE = 'results.pkl'

logger.configure(
  '/home/jonasrothfuss/Dropbox/Eigene_Dateien/Uni/WS17_18/Density_Estimation/Nonparametric_Density_Estimation/data/cluster',
  EXP_PREFIX)

results_from_pkl_file = dict(logger.load_pkl_log(RESULTS_FILE))
gof_result = GoodnessOfFitResults(single_results_dict=results_from_pkl_file)
results_df = gof_result.generate_results_dataframe(base_experiment.KEYS_OF_INTEREST + ['entropy_reg_coef'])

#gof_result = ConfigRunner.load_dumped_estimators(gof_result, task_id=[5])


SMALL_SIZE = 11
MEDIUM_SIZE = 12
LARGE_SIZE = 16
TITLE_SIZE = 20

LINEWIDTH = 6

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=LARGE_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize


""" Entropy Regularization"""
for estimator, n_centers in [("MixtureDensityNetwork", 10), ("KernelMixtureNetwork", 20)]:
  title = "%s (%i kernels) - Entropy Regularization Regularization"%(estimator, n_centers)
  plot_dict = dict([(simulator,
       dict([
         ("entropy_coef=%.3f"%entropy_reg_coef, {"estimator": estimator, "entropy_reg_coef": entropy_reg_coef, "n_centers": n_centers, "simulator": simulator})
              for entropy_reg_coef in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]])
       ) for simulator in ["EconDensity", "ArmaJump", ]])

  fig = gof_result.plot_metric(plot_dict, metric="js_divergence", figsize=(14,6), layout=(1,2))
  plt.suptitle(title, fontsize=TITLE_SIZE)
  plt.tight_layout(h_pad=2, rect=[0, 0, 1, 0.95])
  plt.savefig("%s_%i_entropy_reg.png"%(estimator, n_centers))
  plt.clf()

