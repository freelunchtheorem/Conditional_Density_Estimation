from cde.density_simulation import *
from cde.density_estimator import *
from cde.evaluation.simulation_eval import base_experiment
from ml_logger import logger
from cde.utils.misc import take_of_type
from cde.model_fitting.GoodnessOfFitResults import GoodnessOfFitResults
import cde.model_fitting.ConfigRunner as ConfigRunner
import tensorflow as tf
import os
import pickle
import matplotlib.pyplot as plt




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


def get_density_plots(estimators_list, simulators_dict, path_to_results, exp_prefix="question1_noise_reg_x", task_ids=None):
  """
  This function allows to compare plots from estimators and simulators (i.e. fitted and true densities). Two modes are currently available:
  1) by specifying estimators and simulator, the function picks one result pair randomly that matches the given simulator/estimator
  selection
  2) by specifying the task_ids as list, it is possible to pick specific plots to compare

  Args:
    estimators: a list containing strings of estimators to be evaluated, e.g. ['KernelMixtureNetwork', 'MixtureDensityNetwork']
    simulators: a dict containing specifications of a simulator under which the estimators shall be compared, e.g.
      {'heteroscedastic': True, 'random_seed': 20, 'std': 1, 'simulator': 'EconDensity'}
    path_to_results: absolute path to where the dumped model files are stored
    exp_prefix: specifies the task question

  Returns:
    A list of figures for fitted and true densities.
  """

  if task_ids is not None:
    assert type(task_ids) == list
    assert len(task_ids) == len(estimators_list)


  RESULTS_FILE = 'results.pkl'
  logger.configure(path_to_results, exp_prefix)

  results_from_pkl_file = dict(logger.load_pkl_log(RESULTS_FILE))
  gof_result = GoodnessOfFitResults(single_results_dict=results_from_pkl_file)
  results_df = gof_result.generate_results_dataframe(base_experiment.KEYS_OF_INTEREST)

  """ load model's estimators """
  if task_ids is None:

    models_of_interest = {k: v for k, v in gof_result.single_results_dict.items() if
                          v.probabilistic_model_params == simulators_dict and v.ndim_x + v.ndim_y == 2}

    models = [ConfigRunner.load_dumped_estimator(take_of_type(1, estimator_str, models_of_interest)) for estimator_str in estimators_list]
  else:
    models = [ConfigRunner.load_dumped_estimators(gof_result, task_id=task_ids)]

  """ load model's simulators """
  # todo: implement when simulator dumps exist

  figs = []

  for model in models:
    graph = model.estimator.sess.graph
    sess = tf.Session(graph=graph)

    with sess:
      sess.run(tf.global_variables_initializer())
      model.estimator.sess = sess
      """ fitted density figures"""
      plt.suptitle(model.estimator.name)
      fig_fitted = model.estimator.plot3d()
      figs.append(fig_fitted)

      """ true density figures """
      # todo: use newly dumped simulators

      sess.close()

  return figs


if __name__ == "__main__":
  # simulator = EconDensity(std=1, heteroscedastic=True)
  # estimator = KernelMixtureNetwork('kmn', 1, 1, center_sampling_method='all', n_centers=2000)
  # fit_and_plot_estimated_vs_original_2D(estimator, simulator, 2000)
  #plot_dumped_model('/home/jonasrothfuss/Documents/simulation_eval/question1_noise_reg_x/model_dumps/KernelMixtureNetwork_task_66.pickle')

  #sim = ArmaJump().plot(xlim=(-0.2, 0.2), ylim=(-0.1, 0.1))

  #pickle_path = '/home/jonasrothfuss/Documents/simulation_eval/question1_noise_reg_x/model_dumps/KernelMixtureNetwork_task_66.pickle'


  with open(pickle_path, 'rb') as f:
    with tf.Session() as sess:
      model = pickle.load(f)
      #model.plot2d(x_cond=[0.5, 2.0], ylim=(-4, 8), resolution=200, show=False)

      sim = EconDensity()
      comparison_plot2d_sim_est(model, sim, x_cond=[0.5, 2.0])
