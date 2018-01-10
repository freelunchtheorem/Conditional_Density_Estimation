import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from density_simulation import ConditionalDensity



def econ_conditional_pdf_1_sample(x, n_samples = 1000):
  y = x ** 2 + np.random.normal(loc=0, scale=0.5, size=[n_samples])
  return y

def econ_pdf_1_sample(n_samples = 1000):
  x = np.abs(np.random.standard_normal(size=[n_samples, 1]))
  y = x**2 + np.random.normal(loc=0, scale=0.5, size=[n_samples, 1])
  return np.stack([x,y], axis=1)

def build_econ1_dataset(n_samples=2000):
  sim_data = econ_pdf_1_sample(n_samples=n_samples)
  X_train, X_test, y_train, y_test = train_test_split(sim_data[:,0], sim_data[:,1], random_state=42, train_size=0.6)
  return X_train, X_test, y_train.ravel(), y_test.ravel()

def plot_histogram_pdf_econ1(n_samples=100000):
  sim_data = econ_pdf_1_sample(n_samples=n_samples)
  print(sim_data.shape)

  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  x, y = sim_data[:, 0], sim_data[:, 1]
  hist, xedges, yedges = np.histogram2d(x, y, bins=50, range=[[0, 2], [0, 2]])

  xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25)
  xpos = xpos.flatten('F')
  ypos = ypos.flatten('F')
  zpos = np.zeros_like(xpos)

  # Construct arrays with the dimensions for the 16 bars.
  dx = 0.5 * np.ones_like(zpos)
  dy = dx.copy()
  dz = hist.flatten()

  ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b', zsort='average')
  plt.show()

def plot_histogram_conditional_pdf_econ1(x, n_samples = 100000):
  sim_data = econ_conditional_pdf_1_sample(x, n_samples=n_samples)
  plt.hist(sim_data, bins=100)
  plt.show()
