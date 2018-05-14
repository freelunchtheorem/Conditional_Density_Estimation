from cde.density_simulation import *
from cde.density_estimator import *

import tensorflow as tf
import os
import pickle


def fit_and_plot_estimated_vs_original_2D(estimator, simulator, n_samples):
  X, Y = simulator.simulate(n_samples)
  with tf.Session() as sess:
    estimator.fit(X, Y, verbose=True)
    estimator.plot()


def plot_dumped_model(pickle_path):
  assert os.path.isfile(pickle_path), "pickle path must be file"

  with open(pickle_path, 'rb') as f:
    with tf.Session() as sess:
      model = pickle.load(f)
      model.plot()

if __name__ == "__main__":
  # simulator = EconDensity(std=1, heteroscedastic=True)
  # estimator = KernelMixtureNetwork('kmn', 1, 1, center_sampling_method='all', n_centers=2000)
  # fit_and_plot_estimated_vs_original_2D(estimator, simulator, 2000)
  plot_dumped_model('/home/jonasrothfuss/Dropbox/Eigene_Dateien/Uni/WS17_18/Density_Estimation/Nonparametric_Density_Estimation/cde/evaluation_runs/question1_noise_reg_x/model_dumps/KernelMixtureNetwork_task_4.pickle')