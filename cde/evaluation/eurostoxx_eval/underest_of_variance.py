import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import os
import numpy as np
import tensorflow as tf
import pandas as pd
import itertools
from multiprocessing import Manager

from cde.density_estimator.MDN import MixtureDensityNetwork
from cde.density_estimator.KMN import KernelMixtureNetwork
from cde.density_simulation.LinearGaussian import LinearGaussian
from cde.utils.async_executor import AsyncExecutor


dir_path = os.path.dirname(os.path.realpath(__file__))

seeds = [30, 31, 32, 33, 34, 35, 36, 38, 39, 40]
std = 0.01
x_noise = 0.1
y_noise = 0.1
epochs = 2000
ndims = range(1, 20, 1)

n_workers = 6

manager = Manager()
result_list = manager.list()

def estimate_cov(i, j):
  print('STARTING JOB (%i,%i)'%(i,j))
  ndim_x = ndims[i]
  round_seed = seeds[j]
  with tf.Session() as sess:
    model = LinearGaussian(mu=0, ndim_x=ndim_x, mu_slope=0.0, std=std, std_slope=0.005, random_seed=round_seed)
    X, Y = model.simulate(n_samples=5000)
    ndim_x, ndim_y = X.shape[1], 1

    # fit the estimators
    mdn = MixtureDensityNetwork('mdn_empirical' + str(ndim_x) + str(round_seed), ndim_x, ndim_y, n_centers=20,
                                n_training_epochs=epochs,
                                random_seed=round_seed, x_noise_std=x_noise, y_noise_std=y_noise,
                                data_normalization=True, hidden_sizes=(16, 16))
    mdn.fit(X, Y)

    kmn = KernelMixtureNetwork('kmn_empirical' + str(ndim_x) + str(round_seed), ndim_x, ndim_y, n_centers=50,
                               init_scales=[1.0, 0.5],
                               n_training_epochs=epochs, random_seed=round_seed, x_noise_std=x_noise,
                               y_noise_std=y_noise,
                               data_normalization=True, keep_edges=True, hidden_sizes=(16, 16))
    kmn.fit(X, Y)

    x_cond = np.array([np.zeros(ndim_x)])

    # estimate standard deviation
    std_est_mdn = np.sqrt(mdn.covariance(x_cond)).flatten()
    std_est_kmn = np.sqrt(kmn.covariance(x_cond)).flatten()

    result_list.append(('MDN',i, j, std_est_mdn))
    result_list.append(('KMN', i, j, std_est_kmn))


indices = list(zip(*itertools.product(range(len(ndims)), range(len(seeds)))))
exec = AsyncExecutor(n_jobs=n_workers)
exec.run(estimate_cov, indices[0], indices[1])

result_dict = {'MDN': np.zeros((len(ndims), len(seeds))), 'KMN': np.zeros((len(ndims), len(seeds)))}

for result_tuple in result_list:
  result_dict[result_tuple[0]][result_tuple[1], result_tuple[2]] = result_tuple[3]


# average over seeds
estimated_stds_mdn = np.mean(result_dict['MDN'], axis=-1)
estimated_stds_kmn = np.mean(result_dict['KMN'], axis=-1)

# store results
df = pd.DataFrame(data={'estimated_stds_mdn': estimated_stds_mdn, 'estimated_stds_kmn': estimated_stds_mdn},
                  index=ndims)
df.to_csv(os.path.join(dir_path, 'underest_of_variance.csv'))


# plot the curves
plt.plot(ndims, estimated_stds_mdn, label='estimated_std MDN')
plt.plot(ndims, estimated_stds_kmn, label='estimated_std KMN')
plt.plot(ndims, std * np.ones(len(ndims)), label='true_std')

plt.text(2, 2, 'x_noise = %.3f'%x_noise, fontsize=20)
plt.xlabel('n_dimensions of X')
plt.ylabel('standard deviation')
plt.legend(loc='lower left')

plt.title("Systematic underestimation of variance as dimensionality increases")

plt.savefig(os.path.join(dir_path, 'underest_of_variance.png'))
