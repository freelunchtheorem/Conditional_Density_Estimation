import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from cde.density_estimator.KMN import KernelMixtureNetwork
from matplotlib.lines import Line2D
import pandas as pd

def build_toy_dataset(n_samples=40000):
  y_data = np.float32(np.random.uniform(-10.5, 10.5, (1, n_samples))).T
  r_data = np.float32(np.random.normal(size=(n_samples, 1)))  # random noise
  x_data = np.float32(np.sin(0.75 * y_data) * 7.0 + y_data * 0.5 + r_data * 1.0)
  X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, random_state=42, train_size=0.5)
  return X_train, X_test, y_train.ravel(), y_test.ravel()

def build_toy_dataset2(n_samples=40000):
  # circle shaped density function
  radius = np.float32(np.random.normal(loc=4, scale=1, size=(1, n_samples))).T
  angle = np.float32(np.random.uniform(-np.pi, np.pi, size=(1, n_samples))).T
  x_data = radius * np.cos(angle)
  y_data = radius * np.sin(angle)

  train_size = 0.5
  X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, random_state=42, train_size=train_size, test_size=1-train_size)
  return X_train, X_test, y_train.ravel(), y_test.ravel()

