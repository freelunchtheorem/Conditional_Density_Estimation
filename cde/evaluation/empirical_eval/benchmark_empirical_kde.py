from cde.evaluation.empirical_eval.experiment_util import run_benchmark_train_test_fit_cv, run_benchmark_train_test_fit_cv_ml
import cde.evaluation.empirical_eval.datasets as datasets
from ml_logger import logger
import config
import pandas as pd

EXP_PREFIX = 'benchmark_empirical_kde'


model_dict = {
    'CKDE_cv_ml': {'estimator': 'ConditionalKernelDensityEstimation', 'bandwidth': 'cv_ml'},
    'CKDE_normal_ref': {'estimator': 'ConditionalKernelDensityEstimation', 'bandwidth': 'normal_reference'},
    'NKDE_cv_ml': {'estimator': 'NeighborKernelDensityEstimation', 'param_selection': 'cv_ml'},
    'NKDE_normal_ref': {'estimator': 'NeighborKernelDensityEstimation', 'param_selection': 'normal_reference'},
}

def experiment():
    logger.configure(log_directory=config.DATA_DIR, prefix=EXP_PREFIX, color='green')

    # 1) EUROSTOXX
    dataset = datasets.EuroStoxx50()

    result_df = run_benchmark_train_test_fit_cv_ml(dataset, model_dict, n_train_valid_splits=3, shuffle_splits=False, seed=22)

    # 2)
    for n_samples in [10000]:
        dataset = datasets.NCYTaxiDropoffPredict(n_samples=n_samples)

    df = run_benchmark_train_test_fit_cv_ml(dataset, model_dict, n_train_valid_splits=3, shuffle_splits=True, seed=22)

    result_df = pd.concat([result_df, df], ignore_index=True)

    # 3) UCI & NYC Taxi
    for dataset_class in [datasets.BostonHousing, datasets.Conrete, datasets.Energy]:
        dataset = dataset_class()
        df = run_benchmark_train_test_fit_cv_ml(dataset, model_dict, n_train_valid_splits=3, shuffle_splits=True, seed=22)
        result_df = pd.concat([result_df, df], ignore_index=True)

    logger.log('\n', str(result_df))

if __name__ == "__main__":
    experiment()