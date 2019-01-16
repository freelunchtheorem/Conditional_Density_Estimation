from cde.density_estimator import MixtureDensityNetwork, ConditionalKernelDensityEstimation, \
  NeighborKernelDensityEstimation, LSConditionalDensityEstimation

from cde.density_simulation import LinearGaussian

import numpy as np
import os
from matplotlib import pyplot as plt


DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data'))
SEED = 65

mu = 0.0
std = 1
model = LinearGaussian(ndim_x=1, mu=mu, std=std, mu_slope=0.002, random_seed=SEED)

X, Y = model.simulate(n_samples=4000)

fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(10, 3))

for i, noise_std in enumerate([0.0, 0.05, 0.2]):
  estimator1 = MixtureDensityNetwork('mdn%i'%i, 1, 1, n_centers=20, x_noise_std=noise_std,
                                     y_noise_std=noise_std, random_seed=SEED)
  estimator1.fit(X, Y)


  y = np.linspace(mu-3*std, mu+3*std, num=500)
  x_cond = np.ones(500) * 0.0


  p_true = model.pdf(x_cond, y)
  p_est1 = estimator1.pdf(x_cond, y)

  line1 = axes[i].plot(y, p_true, label='true density', color='green')
  line2 = axes[i].plot(y, p_est1, label='est. density', color='red')
  axes[i].set_title('noise_std = %.2f'%noise_std)


axes[0].set_xlabel('y')
axes[2].set_xlabel('y')
axes[0].set_ylabel('probability density')

fig.legend((line1[0], line2[0]), ('true', 'estimated'), loc='lower center', ncol=2) #loc=(0.857, 0.76))
fig.tight_layout(rect=[0.0, 0.01, 1, 1])

fig_path = os.path.join(DATA_DIR, 'plots/noise_reg/noise_reg_MDN.png')
fig.savefig(fig_path)

fig_path = os.path.join(DATA_DIR, 'plots/noise_reg/noise_reg_MDN.pdf')
fig.savefig(fig_path)

fig.show()

