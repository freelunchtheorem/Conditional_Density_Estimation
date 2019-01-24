from cde.density_simulation import SkewNormal
from cde.density_estimator import KernelMixtureNetwork
import numpy as np

seed = 22

# simulate some data
density_simulator = SkewNormal(random_seed=seed)
X, Y = density_simulator.simulate(n_samples=3000)

# fit density model
model = KernelMixtureNetwork("KDE_demo", ndim_x=1, ndim_y=1, n_centers=50,
                             x_noise_std=0.2, y_noise_std=0.1, random_seed=22)
model.fit(X, Y)

# compute conditional moments & VaR
x_cond = np.zeros((1, 1))

mean = model.mean_(x_cond)[0][0]
std = model.std_(x_cond)[0][0]
skewness = model.skewness(x_cond)[0]
VaR = model.value_at_risk(x_cond, alpha=0.05)[0]

print("Mean:", mean)
print("Std:", std)
print("Skewness:", skewness)
print("Value-at-Risk", VaR)

# plot the fitted distribution
x_cond_plot = np.array([-0.5, 0, 0.5])
model.plot2d(x_cond_plot, ylim=(-0.25, 0.15), show=True)



