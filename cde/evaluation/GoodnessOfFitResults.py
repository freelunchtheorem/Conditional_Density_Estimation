import pandas as pd
import traceback
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

from cde.utils import io


class GoodnessOfFitResults:
  def __init__(self, single_results_dict):
    #assert len(single_results_list) > 0, "given single results list is empty"

    self.single_results_dict = single_results_dict
    self.results_df = None

  def __len__(self):
    return 1

  def generate_results_dataframe(self, keys_of_interest):
    dfs = []
    for single_result in self.single_results_dict.values():
      dfs.append(pd.DataFrame(single_result.report_dict(keys_of_interest=keys_of_interest)))

    self.results_df = pd.concat(dfs, axis=0)
    return self.results_df

  def export_results_as_csv(self, keys_of_interest, output_dir, file_name):
    if self.results_df is None:
      self.generate_results_dataframe(keys_of_interest=keys_of_interest)

    file_results = io.get_full_path(output_dir=output_dir, suffix=".csv", file_name=file_name)
    file_handle_results_csv = open(file_results, "w+")

    """ write result to file"""
    try:
      io.append_result_to_csv(file_handle_results_csv, self.results_df)
    except Exception as e:
      print("exporting results as csv was not successful")
      print(str(e))
      traceback.print_exc()
    finally:
      file_handle_results_csv.close()

  def plot_metric(self, graph_dicts, metric='hellinger_distance', simulator='EconDensity'):
    """
    Generates a plot for a metric with axis x representing the n_observations and y representing the metric.
    Args:

      graph_dicts: a list of dicts, each element representing the data for one curve on the plot, example:
                    graph_dicts = [
                      { "estimator": "KernelMixtureNetwork", "x_noise_std": 0.01, "y_noise_std": 0.01},
                      { ... },
                      ...
                      ]

      metric: must be one of the available metrics (e.g. hellinger_distance, kl_divergence etc.)
      simulator: specifies the simulator, e.g. EconDensity
    """

    assert self.results_df is not None, "first generate results df"
    assert simulator in list(self.results_df["simulator"]), simulator + " not in the results dataframe"
    assert metric in self.results_df
    assert graph_dicts is not None
    assert 'estimator' in self.results_df
    # todo check if estimator and simulator are one of the available

    n_curves_to_plot = len(graph_dicts)
    color = iter(cm.rainbow(np.linspace(0, 1, n_curves_to_plot)))


    d_keys = list(graph_dicts[0].keys())
    d_keys = " ".join(str(x) if x != 'estimator' and x != 'simulator' else "" for x in d_keys)

    for graph_dict in graph_dicts:
      """ data """
      graph_dict['simulator'] = simulator

      sub_df = self.results_df.loc[(self.results_df[list(graph_dict)] == pd.Series(graph_dict)).all(axis=1)]

      metric_values_mean = sub_df.groupby(by='n_observations').mean()[metric]
      metric_values_std = sub_df.groupby(by='n_observations').std()[metric]
      n_obs = metric_values_mean.index

      " visual settings "
      c = next(color)
      label = graph_dict["estimator"] + "_x_noise_" + str(graph_dict["x_noise_std"]) + "_y_noise_" + str(graph_dict["y_noise_std"])
      plt.plot(n_obs, metric_values_mean, color=c, label=label)
      plt.fill_between(n_obs, metric_values_mean - metric_values_std, metric_values_mean + metric_values_std, alpha=0.2, color=c)

    plt.xscale('log')
    plt.xlabel('n_observations')
    plt.ylabel(metric)
    plt.title('Effect of ' + d_keys + " on " + metric)

    plt.legend()
    plt.show()
    print("plot printed")