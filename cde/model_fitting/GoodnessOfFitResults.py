import pandas as pd
import traceback
import numpy as np
import copy


import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from collections import OrderedDict
from cde.utils import io
#from cde.model_fitting.ConfigRunner import ConfigRunner


class GoodnessOfFitResults:
  def __init__(self, single_results_dict):
    #assert len(single_results_list) > 0, "given single results list is empty"

    self.single_results_dict = single_results_dict
    self.results_df = None

  def __len__(self):
    return 1

  def generate_results_dataframe(self, keys_of_interest):
    dfs = []
    for i, single_result in enumerate(self.single_results_dict.values()):

      result_dict = single_result.report_dict(keys_of_interest=keys_of_interest)

      # ndarrays of size > 1 cannot be processed and are replaced by a None
      for key, value in result_dict.items():
        if isinstance(value, np.ndarray) and value.size > 1:
          result_dict[key] = None

      df = pd.DataFrame(result_dict)
      dfs.append(df)

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

  def plot_metric(self, plot_dicts, metric='hellinger_distance', keys_of_interest=None,
                  figsize=(20,8), layout=None, fig=None, color=None, log_scale_x=True, log_scale_y=True):
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
    assert metric in self.results_df
    assert plot_dicts is not None
    assert 'estimator' in self.results_df
    if keys_of_interest is not None:
      assert all(key in self.results_df for key in keys_of_interest), "at least one key of interest not in the results data frame"

    if layout is None:
      layout = (1, len(plot_dicts.keys()))
    if fig is not None:
      axarr = fig.axes
    else:
      fig, axarr = plt.subplots(*layout, figsize=figsize)
      if isinstance(axarr, np.ndarray):
        axarr = axarr.flatten()
      else:
        axarr = np.array([axarr])
    for i, (ax_title, graph_dicts) in enumerate(plot_dicts.items()):

      if color is None:
        color_iter = iter(plt.rcParams['axes.prop_cycle'].by_key()['color'])
      else:
        color_iter = copy.deepcopy(color)


      # d_keys = list(graph_dicts.values()[0].keys())
      # d_keys = " ".join(str(x) if x != 'estimator' and x != 'simulator' else "" for x in d_keys)

      for label, graph_dict in graph_dicts.items():
        """ data """

        sub_df = self.results_df.loc[(self.results_df[list(graph_dict)] == pd.Series(graph_dict)).all(axis=1)]

        metric_values_mean = sub_df.groupby(by='n_observations')[metric].mean()
        metric_values_std = sub_df.groupby(by='n_observations')[metric].std()
        n_obs = metric_values_mean.index


        if keys_of_interest is not None:
          intersect = graph_dict.keys() & keys_of_interest
          intersect.add("estimator")
          intersect.add("simulator")
          sub_dict = OrderedDict((k, graph_dict[k]) for k in intersect)
        else:
          sub_dict = OrderedDict(sorted(graph_dict.items()))

        label = label if label is not None else ', '.join("{}={}".format(k, v) for k, v in sub_dict.items())

        " visual settings "
        c = next(color_iter)

        axarr[i].plot(n_obs, metric_values_mean, color=c, label=label)
        axarr[i].fill_between(n_obs, metric_values_mean - metric_values_std, metric_values_mean + metric_values_std, alpha=0.1, color=c)

      if log_scale_x: axarr[i].set_xscale('log')
      if log_scale_y: axarr[i].set_yscale('log')
      axarr[i].set_xlabel('n_observations')
      axarr[i].set_ylabel(metric)
      axarr[i].set_title(ax_title)
      axarr[i].legend()
    return fig


  def plot_densities(self, selector, configs, metric="hellinger_distance", simulator="EconDensity", mode="pdf", xlim=(-5, 5), ylim=(-5, 5),
                     resolution=100, ):
    assert self.results_df is not None, "first generate results df"
    assert simulator in list(self.results_df["simulator"]), simulator + " not in the results dataframe"
    assert metric in self.results_df
    assert selector


    """ Compares the fitted density (see modes) against the original density

    Args:
      xlim: 2-tuple specifying the x axis limits
      ylim: 2-tuple specifying the y axis limits
      resolution: integer specifying the resolution of plot
      mode: spefify which dist to plot ["pdf", "cdf", "joint_pdf"]

    """
    modes = ["pdf", "cdf", "joint_pdf"]
    assert mode in modes, "mode must be on of the following: " + modes
    #assert self.ndim == 2, "Can only plot two dimensional distributions"

    # prepare mesh
    linspace_x = np.linspace(xlim[0], xlim[1], num=resolution)
    linspace_y = np.linspace(ylim[0], ylim[1], num=resolution)
    X, Y = np.meshgrid(linspace_x, linspace_y)
    X, Y = X.flatten(), Y.flatten()


    selector['simulator'] = 'EconDensity'

    # calculate values of distribution
    if mode == "pdf":
      #task_hash = self.results_df.loc[(self.results_df[list(selector)] == pd.Series(selector)).all(axis=1)].iloc[0]['hash']
      #task = configs[task_hash]

      selected_res = self.results_df.loc[(self.results_df[list(selector)] == pd.Series(selector)).all(axis=1)].iloc[0]
      cfgs_df = pd.DataFrame.from_dict(configs)

      #Z_actual = task['simulator'].pdf(task['X'], task['Y'])
      #Z_recovered = task['estimator'].pdf(task['X'], task['Y'])
    elif mode == "cdf":
      #todo
      Z = self.cdf(X, Y)
    elif mode == "joint_pdf":
      #todo
      Z = self.joint_pdf(X, Y)

    X, Y, Z = X.reshape([resolution, resolution]), Y.reshape([resolution, resolution]), Z.reshape(
      [resolution, resolution])
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, rcount=resolution, ccount=resolution,
                           linewidth=100, antialiased=True)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()






