import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from density_estimator.kmn import KernelMixtureNetwork
from matplotlib.lines import Line2D
import pandas as pd

def build_toy_dataset(n_sample=40000):
  y_data = np.float32(np.random.uniform(-10.5, 10.5, (1, n_sample))).T
  r_data = np.float32(np.random.normal(size=(n_sample, 1)))  # random noise
  x_data = np.float32(np.sin(0.75 * y_data) * 7.0 + y_data * 0.5 + r_data * 1.0)
  X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, random_state=42, train_size=0.5)
  return X_train, X_test, y_train.ravel(), y_test.ravel()

def bild_toy_dataset2(n_sample=40000):
  radius = np.float32(np.random.normal(loc=4, scale=1, size=(1, n_sample))).T
  angle = np.float32(np.random.uniform(-np.pi, np.pi, size=(1, n_sample))).T
  x_data = radius * np.cos(angle)
  y_data = radius * np.sin(angle)

  X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, random_state=42, train_size=0.5)
  return X_train, X_test, y_train.ravel(), y_test.ravel()

n_observations = 2000  # number of data points
n_features = 1  # number of features

X_train, X_test, y_train, y_test = bild_toy_dataset2(n_observations)
print("Size of features in training data: {}".format(X_train.shape))
print("Size of output in training data: {}".format(y_train.shape))
print("Size of features in test data: {}".format(X_test.shape))
print("Size of output in test data: {}".format(y_test.shape))

fig, ax = plt.subplots()
fig.set_size_inches(10, 8)
sns.regplot(X_train, y_train, fit_reg=False);
#plt.savefig('toydata.png')
plt.show()
#plot.figure.size = 100
#plt.show()

kmn = KernelMixtureNetwork(train_scales=True)
kmn.fit(X_train, y_train, n_epoch=300, eval_set=(X_test, y_test))
kmn.plot_loss()

samples = kmn.sample(X_test)
jp = sns.jointplot(X_test.ravel(), samples, kind="hex", stat_func=None, size=10)
jp.ax_joint.add_line(Line2D([X_test[0][0], X_test[0][0]], [-40, 40], linewidth=3))
jp.ax_joint.add_line(Line2D([X_test[1][0], X_test[1][0]], [-40, 40], color='g', linewidth=3))
jp.ax_joint.add_line(Line2D([X_test[2][0], X_test[2][0]], [-40, 40], color='r', linewidth=3))
#plt.savefig('hexplot.png')
plt.show()
d = kmn.predict_density(X_test[0:3,:].reshape(-1,1), resolution=1000)
df = pd.DataFrame(d).transpose()
df.index = np.linspace(kmn.y_min, kmn.y_max, num=1000)
df.plot(legend=False, linewidth=3, figsize=(12.2, 8))
#plt.savefig('conditional_density.png')