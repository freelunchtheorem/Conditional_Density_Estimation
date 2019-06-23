from cde.evaluation.empirical_eval.experiment_util import run_benchmark_train_test_fit_cv
import cde.evaluation.empirical_eval.datasets as datasets
from ml_logger import logger
import config
import pandas as pd

EXP_PREFIX = 'regularization_empirical'


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

x_noise_stds = [0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
y_noise_stds = [0.02, 0.05, 0.1, 0.15, 0.2, 0.25]
l1_reg_params = [0.0, 0.001, 0.01, 0.1, 1.0]
l2_reg_params = [0.0, 0.001, 0.01, 0.1, 1.0]
weight_decay_params = [0.0, 1e-4, 5e-4, 1e-3, 5e-3]

MDN_standard_params = {'estimator': 'MixtureDensityNetwork', 'n_centers': 10, 'n_training_epochs': 1000, 'hidden_sizes': [(32,32)],
                        'weight_normalization': False, 'random_seed': 33}

KMN_standard_params = {'estimator': 'KernelMixtureNetwork', 'n_centers': 50, 'n_training_epochs': 1000,  'hidden_sizes': [(32,32)],
               'weight_normalization': False, 'random_seed': 33}

NF_standard_params = {'estimator': 'NormalizingFlowEstimator', 'n_flows': 10, 'n_training_epochs': 1000,  'hidden_sizes': [(32,32)],
               'weight_normalization': False, 'random_seed': 33}


model_dict = {

    # ---------------- MDN ----------------

    'MDN_cv_noise': {**MDN_standard_params, 'x_noise_std': x_noise_stds, 'y_noise_std': y_noise_stds},

    'MDN_cv_l1': {**MDN_standard_params, 'l1_reg': l1_reg_params},

    'MDN_cv_l2': {**MDN_standard_params, 'l2_reg': l2_reg_params},

    'MDN_cv_weight_decay': {**MDN_standard_params, 'weight_decay': weight_decay_params},


    # ---------------- KMN ----------------

    'KMN_cv_noise': {**KMN_standard_params, 'x_noise_std': x_noise_stds, 'y_noise_std': y_noise_stds},
    'KMN_cv_l1': {**KMN_standard_params, 'l1_reg': l1_reg_params},
    'KMN_cv_l2': {**KMN_standard_params, 'l2_reg': l2_reg_params},
    'KMN_cv_weight_decay': {**KMN_standard_params, 'weight_decay': weight_decay_params},

    # ---------------- NF ------------------

    'NF_cv_noise': {**NF_standard_params, 'x_noise_std': x_noise_stds, 'y_noise_std': y_noise_stds},
    'NF_cv_l1': {**NF_standard_params, 'l1_reg': l1_reg_params},
    'NF_cv_l2': {**NF_standard_params, 'l2_reg': l2_reg_params},
    'NF_cv_weight_decay': {**NF_standard_params, 'weight_decay': weight_decay_params},

}

def experiment():
    logger.configure(log_directory=config.DATA_DIR, prefix=EXP_PREFIX, color='green')

    # 1) EUROSTOXX
    dataset = datasets.EuroStoxx50()

    result_df = run_benchmark_train_test_fit_cv(dataset, model_dict, n_train_valid_splits=3, n_eval_seeds=5, shuffle_splits=False,
                                    n_folds=5, seed=22, n_jobs_inner=-1, n_jobc_outer=3)

    # 2) NYC Taxi
    for n_samples in [10000]:
        dataset = datasets.NCYTaxiDropoffPredict(n_samples=n_samples)

    df = run_benchmark_train_test_fit_cv(dataset, model_dict, n_train_valid_splits=3, n_eval_seeds=5, shuffle_splits=True,
                                    n_folds=5, seed=22, n_jobs_inner=-1, n_jobc_outer=3)
    result_df = pd.concat([result_df, df], ignore_index=True)

    # 3) UCI
    for dataset_class in [datasets.BostonHousing, datasets.Conrete, datasets.Energy]:
        dataset = dataset_class()
        df = run_benchmark_train_test_fit_cv(dataset, model_dict, n_train_valid_splits=3, n_eval_seeds=5,
                                        shuffle_splits=True, n_folds=5, seed=22, n_jobs_inner=-1, n_jobc_outer=3)
        result_df = pd.concat([result_df, df], ignore_index=True)

    logger.log('\n', str(result_df))
    logger.log('\n', result_df.tolatex())

if __name__ == "__main__":
    experiment()