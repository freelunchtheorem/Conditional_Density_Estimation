import numpy as np


class GoodnessOfFitResults:
  def __init__(self, x_cond, estimator, probabilistic_model):
    self.cond_values = x_cond

    self.time_to_fit = None
    self.time_to_predict = None

    self.ndim_x = estimator.ndim_x
    self.ndim_y = estimator.ndim_y

    self.estimator_params = estimator.get_params()
    self.probabilistic_model_params = probabilistic_model.get_params()

    self.kl_divergence = None
    self.hellinger_distance = None
    self.wasserstein_distance = None
    self.js_divergence = None

    self.n_observations = None


  def report_dict(self):
    full_dict = self.__dict__
    keys_of_interest = ["n_observations", "ndim_x", "ndim_y", "kl_divergence", "hellinger_distance", "js_divergence", "time_to_fit",
                        "time_to_predict"]
    report_dict = dict([(key, full_dict[key]) for key in keys_of_interest])

    get_from_dict = lambda key: self.estimator_params[key] if key in self.estimator_params else None

    for key in ["estimator", "n_centers", "center_sampling_method"]:
      report_dict[key] = get_from_dict(key)


    report_dict["simulator"] = self.probabilistic_model_params["simulator"]

    return report_dict

  def __len__(self):
    return 1


  def __str__(self):
    return "KL divergence: %.4f, Hellinger distance: %.4f, Jason-Shannon divergence: %.4f"%(self.kl_divergence, self.hellinger_distance, self.js_divergence)