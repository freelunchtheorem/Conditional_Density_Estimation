import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from density_estimator.kmn import KernelMixtureNetwork
from matplotlib.lines import Line2D
import pandas as pd
from density_simulation.econ_densities import build_econ1_dataset
from density_simulation.toy_densities import build_toy_dataset, bild_toy_dataset2



def main():
  n_observations = 2000  # number of data points
  n_features = 1  # number of features

  X_train, X_test, y_train, y_test = bild_toy_dataset2(n_observations)
  print("Size of features in training data: {}".format(X_train.shape))
  print("Size of output in training data: {}".format(y_train.shape))
  print("Size of features in test data: {}".format(X_test.shape))
  print("Size of output in test data: {}".format(y_test.shape))

  fig, ax = plt.subplots()
  fig.set_size_inches(10, 8)
  sns.regplot(X_train, y_train, fit_reg=False)
  plt.savefig('toydata.png')
  plt.show()
  # plot.figure.size = 100
  # plt.show()

  kmn = KernelMixtureNetwork(train_scales=True, n_centers=10)
  kmn.fit(X_train, y_train, n_epoch=300, eval_set=(X_test, y_test))
  kmn.plot_loss()
  plt.savefig('trainplot.png')

  samples = kmn.sample(X_test)
  jp = sns.jointplot(X_test.ravel(), samples, kind="hex", stat_func=None, size=10)
  jp.ax_joint.add_line(Line2D([X_test[0][0], X_test[0][0]], [-40, 40], linewidth=3))
  jp.ax_joint.add_line(Line2D([X_test[1][0], X_test[1][0]], [-40, 40], color='g', linewidth=3))
  jp.ax_joint.add_line(Line2D([X_test[2][0], X_test[2][0]], [-40, 40], color='r', linewidth=3))
  plt.savefig('hexplot.png')
  plt.show()
  d = kmn.predict_density(np.expand_dims(np.asarray([0.5,1,2]), axis=1).reshape(-1, 1), resolution=1000)
  df = pd.DataFrame(d).transpose()
  df.index = np.linspace(kmn.y_min, kmn.y_max, num=1000)
  df.plot(legend=False, linewidth=3, figsize=(12.2, 8))
  plt.savefig('conditional_density.png')

if __name__ == "__main__":
  main()