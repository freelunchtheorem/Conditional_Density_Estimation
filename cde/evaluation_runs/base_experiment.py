import argparse

from cde.evaluation.ConfigRunner import ConfigRunner


RESULTS_FILE = 'results.pkl'

KEYS_OF_INTEREST = ['task_name', 'estimator', 'simulator', 'n_observations', 'center_sampling_method', 'x_noise_std',
                      'y_noise_std', 'ndim_x', 'ndim_y', 'n_centers', "n_mc_samples", "n_x_cond", 'mean_est',
                      'cov_est', 'mean_sim', 'cov_sim', 'kl_divergence', 'hellinger_distance', 'js_divergence',
                      'x_cond', 'random_seed', "mean_sim", "cov_sim", "mean_abs_diff", "cov_abs_diff",
                      "VaR_sim", "VaR_est", "VaR_abs_diff", "CVaR_sim", "CVaR_est", "CVaR_abs_diff",
                      "time_to_fit"
                      ]


def launch_experiment(conf_est, conf_sim, observations, exp_prefix):

  parser = argparse.ArgumentParser(description='Run configuration script')
  parser.add_argument('--parallel', type=bool, default=True,
                      help='an integer for the accumulator')
  parser.add_argument('--n_workers', type=int, default=1,
                      help='sum the integers (default: find the max)')
  parser.add_argument('--use_gpu', type=bool, default=False)
  parser.add_argument('--load', type=bool, default=True)
  args = parser.parse_args()


  conf_runner = ConfigRunner(exp_prefix, conf_est, conf_sim, observations=observations, keys_of_interest=KEYS_OF_INTEREST,
                             n_mc_samples=int(2e6), n_x_cond=10, n_seeds=5, use_gpu=args.use_gpu)

  conf_runner.run_configurations(dump_models=True, multiprocessing=args.parallel, n_workers=args.n_workers)

  return args.load