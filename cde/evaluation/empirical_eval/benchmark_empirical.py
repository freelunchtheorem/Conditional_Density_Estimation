from cde.evaluation.empirical_eval.experiment_util import run_benchmark_train_test_fit_cv, run_benchmark_train_test_fit_cv_ml
import cde.evaluation.empirical_eval.datasets as datasets
from ml_logger import logger
import config
import pandas as pd

EXP_PREFIX = 'benchmark_empirical'


class Rule_of_thumb:

    def __init__(self, scale_factor):
        self.scale_factor = scale_factor

    def __call__(self, n, d):
        return self.scale_factor * n ** (-1 / (4 + d))

    def __str__(self):
        return "rule_of_thumb_%.2f" % self.scale_factor


class Polynomial_Rate:
    def __init__(self, scale_factor, order):
        self.scale_factor = scale_factor
        self.order = order

    def __call__(self, n, d):
        return self.scale_factor * n ** (-1 / (self.order + d))

    def __str__(self):
        return "polynomial_rate_%i_%.2f" % (self.order, self.scale_factor)


# setup model dict
adaptive_noise_functions = [Rule_of_thumb(1.0), Rule_of_thumb(0.7), Rule_of_thumb(0.5),
                            Polynomial_Rate(2.0, 1), Polynomial_Rate(1.0, 1), Polynomial_Rate(1.0, 2),
                            Polynomial_Rate(2.0, 2), Polynomial_Rate(1.0, 3), Polynomial_Rate(2.0, 3)]

x_noise_stds = [0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
y_noise_stds = [0.02, 0.05, 0.1, 0.15, 0.2, 0.25]

MDN_standard_params = {'estimator': 'MixtureDensityNetwork', 'n_training_epochs': 1000, 'hidden_sizes': [(32,32)],
                        'weight_normalization': False, 'random_seed': 40}

KMN_standard_params = {'estimator': 'KernelMixtureNetwork', 'n_training_epochs': 1000,  'hidden_sizes': [(32,32)],
               'weight_normalization': False, 'random_seed': 40}

NF_standard_params = {'estimator': 'NormalizingFlowEstimator', 'n_training_epochs': 1000,  'hidden_sizes': [(32,32)],
               'weight_normalization': False, 'random_seed': 40}


model_dict = {

    # ---------------- MDN ----------------

    'MDN_cv': {**MDN_standard_params, 'x_noise_std': x_noise_stds, 'y_noise_std': y_noise_stds, 'dropout': [0.0, 0.2],
                              'n_centers': [5, 10, 20, 50]},

    # ---------------- KMN ----------------

    'KMN_cv': {**KMN_standard_params, 'x_noise_std': x_noise_stds, 'y_noise_std': y_noise_stds, 'dropout': [0.0, 0.2],
                              'n_centers': [20, 50, 200]},
    # ---------------- NF ------------------

    'NF_cv': {**NF_standard_params, 'x_noise_std': x_noise_stds, 'y_noise_std': y_noise_stds, 'dropout': [0.0, 0.2],
                             'n_flows': [5, 10, 20, 50]},

    'LSCDE_cv': {'estimator': 'LSConditionalDensityEstimation', 'bandwidth': [0.1, 0.2, 0.5, 0.7],
             'n_centers': [500, 1000], 'regularization': [0.1, 0.5, 1.0, 4.0, 8.0], 'random_seed': 40},
}

def experiment():
    logger.configure(log_directory=config.DATA_DIR, prefix=EXP_PREFIX, color='green')

    # 1) EUROSTOXX
    dataset = datasets.EuroStoxx50()

    result_df = run_benchmark_train_test_fit_cv(dataset, model_dict, n_train_valid_splits=3, n_eval_seeds=5, shuffle_splits=False,
                                    n_folds=5, seed=22)

    # 2) NYC Taxi
    for n_samples in [10000]:
        dataset = datasets.NCYTaxiDropoffPredict(n_samples=n_samples)

    df = run_benchmark_train_test_fit_cv(dataset, model_dict, n_train_valid_splits=3, n_eval_seeds=5, shuffle_splits=True,
                                    n_folds=5, seed=22,  n_jobs_inner=-1, n_jobc_outer=2)
    result_df = pd.concat([result_df, df], ignore_index=True)


    # 3) UCI
    result_df = None
    for dataset_class in [datasets.BostonHousing, datasets.Conrete, datasets.Energy]:
        dataset = dataset_class()
        df = run_benchmark_train_test_fit_cv(dataset, model_dict, n_train_valid_splits=1, n_eval_seeds=5,
                                             shuffle_splits=True, n_folds=5, seed=22, n_jobs_inner=-1,
                                             n_jobc_outer=2)
        result_df = pd.concat([result_df, df], ignore_index=True)

    logger.log('\n', str(result_df))

if __name__ == "__main__":
    experiment()