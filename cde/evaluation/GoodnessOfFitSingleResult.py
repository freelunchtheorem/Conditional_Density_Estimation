from collections import OrderedDict

class GoodnessOfFitSingleResult:
  def __init__(self, x_cond, estimator_params, probabilistic_model_params):
    self.cond_values = x_cond

    self.time_to_fit = None
    self.time_to_predict = None

    self.ndim_x = estimator_params["ndim_x"]
    self.ndim_y = estimator_params["ndim_y"]

    # remove built in functions so that GoodnessOfFitSingleResult remains pickable
    if 'X_ph' in estimator_params:
      del estimator_params['X_ph']

    self.estimator_params = estimator_params
    self.probabilistic_model_params = probabilistic_model_params

    self.kl_divergence = None
    self.hellinger_distance = None
    self.wasserstein_distance = None
    self.js_divergence = None

    self.n_observations = None

    self.x_cond = x_cond
    self.n_x_cond = len(x_cond)
    self.result_df = None
    self.random_seed = probabilistic_model_params['random_seed']


  def report_dict(self, keys_of_interest=None):
    full_dict = self.__dict__

    if keys_of_interest is not None:
      report_dict = OrderedDict()
      for key in keys_of_interest:
        if key in full_dict:
          report_dict[key] = full_dict[key]
        elif key in self.estimator_params:
          report_dict[key] = self.estimator_params[key]
        elif key in self.probabilistic_model_params:
          report_dict[key] = self.probabilistic_model_params[key]
        else:
          report_dict[key] = None

      return report_dict
    else:
      return full_dict

  def __len__(self):
    return 1


  def __str__(self):
    return "KL divergence: %.4f, Hellinger distance: %.4f, Jason-Shannon divergence: %.4f"%(self.kl_divergence, self.hellinger_distance, self.js_divergence)
