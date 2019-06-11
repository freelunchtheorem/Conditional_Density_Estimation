import argparse

from cde.model_fitting.ConfigRunner import ConfigRunner
from cde.model_fitting.ConfigRunnerLogProb import ConfigRunnerLogProb


RESULTS_FILE = 'results.pkl'

KEYS_OF_INTEREST = [
    'task_name', 'estimator', 'simulator', 'n_observations', 'center_sampling_method', 'x_noise_std',
    'y_noise_std', 'ndim_x', 'ndim_y', 'flows_type', 'n_centers', 'dropout', 'weight_normalization', 'weight_decay',
    "n_mc_samples", "n_x_cond", 'mean_est',
    'std_est', 'mean_sim', 'std_sim', 'kl_divergence', 'hellinger_distance', 'js_divergence',
    'x_cond', 'random_seed', "mean_abs_diff", "std_abs_diff",
    "time_to_fit"
]

KEYS_OF_INTEREST_TAILS = ["VaR_sim", "VaR_est", "VaR_abs_diff", "CVaR_sim", "CVaR_est", "CVaR_abs_diff"]

KEYS_OF_INTEREST_LOGPROB = [
    'task_name', 'estimator', 'simulator', 'n_observations', 'center_sampling_method', 'x_noise_std',
    'y_noise_std', 'ndim_x', 'ndim_y', 'flows_type', 'n_centers', 'dropout', 'weight_normalization', 'weight_decay',
    'random_seed', 'time_to_fit', 'score', 'adaptive_noise_fn', 'l1_reg', 'l2_reg'
]


def launch_experiment(conf_est, conf_sim, observations, exp_prefix, n_mc_samples=10**6, n_x_cond=10, n_seeds=5, tail_measures=False):
    """
    :param conf_est: Dict with keys: Name of estimator, value: params for the estimator
    :param conf_sim: Dict with keys: Name of the density simulator, value: params for the simulator
    :param observations: List or scalar that defines how many samples to use from the distribution
    :param exp_prefix: directory to save everything
    :param n_mc_samples: number of samples to use for Monte Carlo sampling
    :param n_x_cond: number of x conditionals to be sampled
    :param n_seeds: number of seeds to use for the simulator
    :param tail_measures:
    :return:
    """

    parser = argparse.ArgumentParser(description='Run configuration script')
    parser.add_argument('--parallel', type=bool, default=True)
    parser.add_argument('--n_workers', type=int, default=1)
    parser.add_argument('--use_gpu', type=bool, default=False)
    parser.add_argument('--load', type=bool, default=True)
    args = parser.parse_args()

    if tail_measures:
        KEYS_OF_INTEREST.append(KEYS_OF_INTEREST_TAILS)

    conf_runner = ConfigRunner(exp_prefix, conf_est, conf_sim, observations=observations, keys_of_interest=KEYS_OF_INTEREST,
                               n_mc_samples=n_mc_samples, n_x_cond=n_x_cond, n_seeds=n_seeds, use_gpu=args.use_gpu,
                               tail_measures=tail_measures)

    conf_runner.run_configurations(dump_models=True, multiprocessing=args.parallel, n_workers=args.n_workers)

    return args.load


def launch_logprob_experiment(conf_est, conf_sim, observations, exp_prefix, n_test_samples=10**5, n_seeds=5):
    """
    :param conf_est: Dict with keys: Name of estimator, value: params for the estimator
    :param conf_sim: Dict with keys: Name of the density simulator, value: params for the simulator
    :param observations: List or scalar that defines how many samples to use from the distribution
    :param exp_prefix: directory to save everything
    :param n_test_samples: number of samples samples to compute the test scroe (lop-probability)
    :param n_seeds: number of seeds to use for the simulator
    :return:
    """

    parser = argparse.ArgumentParser(description='Run configuration script')
    parser.add_argument('--parallel', type=bool, default=True)
    parser.add_argument('--n_workers', type=int, default=1)
    parser.add_argument('--use_gpu', type=bool, default=False)
    parser.add_argument('--load', type=bool, default=True)
    args = parser.parse_args()


    conf_runner = ConfigRunnerLogProb(exp_prefix, conf_est, conf_sim, observations=observations, keys_of_interest=KEYS_OF_INTEREST_LOGPROB,
                               n_test_samples=n_test_samples, n_seeds=n_seeds, use_gpu=args.use_gpu)

    conf_runner.run_configurations(dump_models=False, multiprocessing=args.parallel, n_workers=args.n_workers)

    return args.load