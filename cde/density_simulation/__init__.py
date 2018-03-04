from cde.density_simulation.Density import ConditionalDensity
from cde.density_simulation.GMM import GaussianMixture
from cde.density_simulation.EconDensity import EconDensity
import inspect, sys
import numpy as np

def get_probabilistic_models_list():
  clsmembers = inspect.getmembers(sys.modules[__name__], inspect.isclass)
  return np.asarray(clsmembers)

