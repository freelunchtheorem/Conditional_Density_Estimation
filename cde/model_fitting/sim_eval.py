import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from cde.density_estimator import LSConditionalDensityEstimation, NeighborKernelDensityEstimation, KernelMixtureNetwork
from matplotlib.lines import Line2D
import pandas as pd
from cde.density_simulation import GaussianMixture, EconDensity
from cde.model_fitting.GoodnessOfFit import GoodnessOfFit
from cde.density_simulation.toy_densities import build_toy_dataset, build_toy_dataset2
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def generate_report():
  econ_density = EconDensity()

  X, Y = econ_density.simulate(n_samples=1000)
  nke = NeighborKernelDensityEstimation()
  nke.fit_by_cv(X, Y)

  n_samples = 500
  X_test = np.asarray([1 for _ in range(n_samples)])
  Y_test = np.linspace(0, 8, num=n_samples)
  Z = nke.pdf(X_test, Y_test)



def eval_econ_data():
  gmm = GaussianMixture(ndim_x=1, ndim_y=1)
  econ_density = EconDensity()

  # print("ECON DATA --------------")
  # print("KMN")
  # for n_centers in [50, 100, 200]:
  #   kmn = KernelMixtureNetwork(n_centers=n_centers)
  #   gof = GoodnessOfFit(kmn, econ_density, n_observations=2000, print_fit_result=False, repeat_kolmogorov=1)
  #   gof_results = gof.compute_results()
  #   print("N_Centers:", n_centers)
  #   print(gof_results)

  print("LAZY-Learner:")
  nkde = KernelMixtureNetwork(n_training_epochs=10)
  gof = GoodnessOfFit(nkde, gmm, n_observations=100, print_fit_result=False)
  gof_results = gof.compute_results()
  print(gof_results)
  print(gof_results.report_dict())

  #
  # print("LSCDE")
  # lscde = LSConditionalDensityEstimation()
  # X, Y = econ_density.simulate(2000)
  # nkde.fit_by_cv(X,Y)
  # gof = GoodnessOfFit(lscde, econ_density, n_observations=2000, print_fit_result=False)
  # gof_results = gof.compute_results()
  # print(gof_results)




def plot_fitted_distribution():
  n_observations = 1000  # number of data points
  n_features = 3  # number of features

  np.random.seed(22)


  X_train, X_test, Y_train, Y_test = econ_density.simulate(n_observations)
  model = KernelMixtureNetwork()

  X_train = np.random.normal(loc=0, size=[n_observations, 1])
  Y_train = 3 * X_train + np.random.normal(loc=0, size=[n_observations, 1])
  X_test = np.random.normal(loc=0, size=[100, 1])
  Y_test = 3 * X_test + np.random.normal(loc=0, size=[100, 1])

  model.fit(X_train, Y_train)
  print(model.score(X_test, Y_test))
  #print(model.fit_by_cv(X_train, Y_train))



  # plt.scatter(model.X_train, model.Y_test)
  # plt.scatter(model.centr_x, model.centr_y, s=10*model.alpha)
  # plt.show()
  #
  # fig, ax = plt.subplots()
  # fig.set_size_inches(10, 8)
  # sns.regplot(X_train, Y_train, fit_reg=False)
  # plt.show()
  #
  #


  n_samples = 1000

  Y_plot = np.linspace(-10, 10, num=n_samples)

  X_plot = np.expand_dims(np.asarray([-1 for _ in range(n_samples)]), axis=1)
  result = model.pdf(X_plot, Y_plot)
  plt.plot(Y_plot, result)
  #plt.show()

  #2d plot
  X_plot = np.expand_dims(np.asarray([2 for _ in range(n_samples)]), axis=1)
  result = model.pdf(X_plot, Y_plot)
  plt.plot(Y_plot, result)

  plt.show()

  #3d plot
  n_samples = 100
  linspace_x = np.linspace(-15, 15, num=n_samples)
  linspace_y = np.linspace(-15, 15, num=n_samples)
  X, Y = np.meshgrid(linspace_x, linspace_y)
  X, Y = X.flatten(), Y.flatten()

  Z = model.pdf(X, Y)

  X, Y, Z = X.reshape([n_samples, n_samples]), Y.reshape([n_samples, n_samples]), Z.reshape([n_samples, n_samples])
  fig = plt.figure()
  ax = fig.gca(projection='3d')
  surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                         linewidth=0, antialiased=True)

  plt.show()



def eval1():
  n_observations = 2000  # number of data points
  n_features = 1  # number of features

  X_train, X_test, y_train, y_test = build_econ1_dataset(n_observations)
  print("Size of features in training data: {}".format(X_train.shape))
  print("Size of output in training data: {}".format(y_train.shape))
  print("Size of features in test data: {}".format(X_test.shape))
  print("Size of output in test data: {}".format(y_test.shape))

  fig, ax = plt.subplots()
  fig.set_size_inches(10, 8)
  sns.regplot(X_train, y_train, fit_reg=False)
  # plt.savefig('toydata.png')
  # plt.show()
  # plot.figure.size = 100
  # plt.show()

  kmn = KernelMixtureNetwork(train_scales=True, n_centers=20)
  kmn.fit(X_train, y_train, n_epoch=300, eval_set=(X_test, y_test))
  kmn.plot_loss()
  # plt.savefig('trainplot.png')
  samples = kmn.sample(X_test)
  print(X_test.shape, samples.shape)
  jp = sns.jointplot(X_test.ravel(), samples, kind="hex", stat_func=None, size=10)
  jp.ax_joint.add_line(Line2D([X_test[0][0], X_test[0][0]], [-40, 40], linewidth=3))
  jp.ax_joint.add_line(Line2D([X_test[1][0], X_test[1][0]], [-40, 40], color='g', linewidth=3))
  jp.ax_joint.add_line(Line2D([X_test[2][0], X_test[2][0]], [-40, 40], color='r', linewidth=3))
  plt.savefig('hexplot.png')
  plt.show()
  d = kmn.predict_density(X_test[0:3, :].reshape(-1, 1), resolution=1000)
  df = pd.DataFrame(d).transpose()
  df.index = np.linspace(kmn.y_min, kmn.y_max, num=1000)
  df.plot(legend=False, linewidth=3, figsize=(12.2, 8))
  plt.savefig('conditional_density.png')




def main():
  #test_nkde()
  eval_econ_data()
  #plot_fitted_distribution()
  #eval1()





if __name__ == "__main__":
  main()