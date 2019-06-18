from multiprocessing import Manager
from cde.utils.async_executor import AsyncExecutor, LoopExecutor
import numpy as np
import tensorflow as tf
import pandas as pd
import copy
from pprint import pprint
from ml_logger import logger

from cde.density_estimator import LSConditionalDensityEstimation, KernelMixtureNetwork, MixtureDensityNetwork, \
    ConditionalKernelDensityEstimation, NeighborKernelDensityEstimation, NormalizingFlowEstimator

from cde.evaluation.empirical_eval.datasets import BostonHousing

def run_benchmark_train_test_fit_cv(dataset, model_dict, seed=27, n_jobs_inner=-1, n_jobc_outer=1, n_train_valid_splits=1,
                                    shuffle_splits=True, n_eval_seeds=1, n_folds=5):

    assert shuffle_splits or n_train_valid_splits == 1, "if not shuffling splits, the number of different splits must be 1"

    if logger.log_directory is None:
        logger.configure(log_directory='/tmp/ml-logger')

    logger.log("\n------------------  empirical benchmark with %s ----------------------"%str(dataset))

    rds = np.random.RandomState(seed)
    eval_seeds = list(rds.randint(0, 10**7, size=n_eval_seeds))

    for model_key in model_dict:
        model_dict[model_key].update({'ndim_x': dataset.ndim_x, 'ndim_y': dataset.ndim_y})

    # run experiments
    cv_result_dicts = []
    for i in range(n_train_valid_splits):
        logger.log("--------  train-valid split %i --------"%i)

        X_train, Y_train, X_valid, Y_valid = dataset.get_train_valid_split(valid_portion=0.2, shuffle=True,
                                                                           random_state=rds)

        manager = Manager()
        cv_result_dict = manager.dict()

        def _fit_by_cv_and_eval(estimator_key, conf_dict):
            estimator, param_grid, param_dict_init = _initialize_model(estimator_key, conf_dict, verbose=True)

            # 1) perform cross-validation hyperparam search to select params
            selected_params = estimator.fit_by_cv(X_train, Y_train, param_grid=param_grid, n_folds=n_folds,
                                                  n_jobs=n_jobs_inner, random_state=rds)

            logger.log("%s selected params:"%estimator_key)
            logger.log_params(**selected_params)
            # 2) evaluate selected params with different initializations
            param_dict_init.update(selected_params)

            logger.log("evaluating %s parameters with %i seeds"%(estimator_key, len(eval_seeds)))
            scores = _evaluate_params(estimator.__class__, param_dict_init, X_train, Y_train, X_valid, Y_valid,
                                      seeds=eval_seeds)

            cv_result_dict[estimator_key] = {'selected_params': selected_params, 'scores': scores, 'eval_seeds': eval_seeds}
            logger.log("evaluation scores: %s" % str (scores))


        executor = AsyncExecutor(n_jobs=n_jobc_outer)
        executor.run(_fit_by_cv_and_eval, model_dict.keys(), model_dict.values())

        cv_result_dicts.append(dict(cv_result_dict))


    pprint(cv_result_dicts)

    # rearrange results as pandas df
    final_results_dict = {'scores_mean':[], 'scores_std':[], 'dataset':[]}
    for estimator_key in model_dict.keys():
        scores = []
        for result_dict in cv_result_dicts:
            scores.extend(result_dict[estimator_key]['scores'])

        final_results_dict['scores_mean'].append(np.mean(scores))
        final_results_dict['scores_std'].append(np.std(scores))
        final_results_dict['dataset'].append(str(dataset))

    df = pd.DataFrame.from_dict(data=final_results_dict, orient='columns')
    df.index = list(model_dict.keys())

    logger.log('\n' + str(df))
    return df

""" helpers """

def _initialize_model(model_key, conf_dict, verbose=False):
    ''' make kartesian product of listed parameters per model '''
    assert 'estimator' in conf_dict.keys()
    estimator = conf_dict.pop('estimator')
    param_dict_cv = {}
    param_dict_init = {}
    for param_key, param_value in conf_dict.items():
        if type(param_value) in (list, tuple):
            param_dict_cv[param_key] = param_value
            param_dict_init[param_key] = param_value[0]
        else:
            param_dict_init[param_key] = param_value

    param_dict_init['name'] = model_key

    if verbose: logger.log('initialize %s'%model_key)

    estimator_instance = globals()[estimator](**param_dict_init)

    return estimator_instance, param_dict_cv, param_dict_init

def _evaluate_params(estimator_class, param_dict, X_train, Y_train, X_valid, Y_valid, seeds):
        eval_scores = []

        for seed in seeds:
            with tf.Session():
                param_dict_local = copy.copy(param_dict)
                param_dict_local['random_seed'] = seed
                param_dict_local['name'] += str(seed)
                est = estimator_class(**param_dict_local)
                est.fit(X_train, Y_train, verbose=False)
                score = est.score(X_valid, Y_valid)
                eval_scores.append(score)

        return eval_scores





