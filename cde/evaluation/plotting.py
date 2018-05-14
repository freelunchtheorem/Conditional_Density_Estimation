from cde.density_simulation import *
from cde.density_estimator import *

import tensorflow as tf
import os
import pickle

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


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
      model.plot2d(x_cond=[0.5, 2.0], ylim=(-4,8), resolution=200, show=False)


def comparison_plot2d_sim_est(est, sim, x_cond=[1.0, 2.0], ylim=(-4,8), resolution=200, mode='pdf'):
  fig, ax = plt.subplots()
  legend_entries = []
  for i in range(len(x_cond)):
    Y = np.linspace(ylim[0], ylim[1], num=resolution)
    X = np.array([x_cond[i] for _ in range(resolution)])
    # calculate values of distribution

    print(X.shape, Y.shape)
    if mode == "pdf":
      Z_est = est.pdf(X, Y)
      Z_sim = sim.pdf(X, Y)
    elif mode == "cdf":
      Z_est = est.cdf(X, Y)
      Z_sim = sim.pdf(X, Y)

    ax.plot(Y, Z_est, label='est ' + 'x=%.2f' % x_cond[i])
    legend_entries.append('est ' + "x=%.2f"%x_cond[i])
    ax.plot(Y, Z_sim, label='sim ' + 'x=%.2f' % x_cond[i])
    legend_entries.append('sim ' + "x=%.2f" % x_cond[i])

  plt.legend(legend_entries,  loc='upper right')

  plt.xlabel("y")
  plt.ylabel("p(y|x)")
  plt.show()


if __name__ == "__main__":
  # simulator = EconDensity(std=1, heteroscedastic=True)
  # estimator = KernelMixtureNetwork('kmn', 1, 1, center_sampling_method='all', n_centers=2000)
  # fit_and_plot_estimated_vs_original_2D(estimator, simulator, 2000)
  #plot_dumped_model('/home/jonasrothfuss/Documents/evaluation_runs/question1_noise_reg_x/model_dumps/KernelMixtureNetwork_task_66.pickle')

  #sim = ArmaJump().plot(xlim=(-0.2, 0.2), ylim=(-0.1, 0.1))

  #pickle_path = '/home/jonasrothfuss/Documents/evaluation_runs/question1_noise_reg_x/model_dumps/KernelMixtureNetwork_task_66.pickle'


  with open(pickle_path, 'rb') as f:
    with tf.Session() as sess:
      model = pickle.load(f)
      #model.plot2d(x_cond=[0.5, 2.0], ylim=(-4, 8), resolution=200, show=False)

      sim = EconDensity()
      comparison_plot2d_sim_est(model, sim, x_cond=[0.5, 2.0])
