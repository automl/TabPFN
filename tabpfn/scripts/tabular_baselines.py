import pandas
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import ParameterGrid

import tempfile
import random
import math
import os
#from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
from sklearn import preprocessing
from torch import nn
from sklearn.metrics import make_scorer
from sklearn.impute import SimpleImputer


from sklearn.base import BaseEstimator, ClassifierMixin
import xgboost as xgb
from sklearn import neighbors
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
import numpy as np
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning) 

import torch
import itertools
from tabpfn.scripts import tabular_metrics
import pandas as pd
from tqdm import tqdm
from tabpfn.utils import remove_outliers

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score
import time

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials , space_eval, rand
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

CV = 5
MULTITHREAD = -1 # Number of threads baselines are able to use at most
param_grid, param_grid_hyperopt = {}, {}

def get_scoring_direction(metric_used):
    # Not needed
    if metric_used.__name__ == tabular_metrics.auc_metric.__name__:
        return -1
    elif metric_used.__name__ == tabular_metrics.cross_entropy.__name__:
        return 1
    else:
        raise Exception('No scoring string found for metric')

def is_classification(metric_used):
    if metric_used.__name__ == tabular_metrics.auc_metric.__name__ or metric_used.__name__ == tabular_metrics.cross_entropy.__name__:
        return 'classification'
    elif metric_used.__name__ == tabular_metrics.auc_metric.__name__:
        return -1

# Loss

def get_scoring_string(metric_used, multiclass=True, usage="sklearn_cv"):
    if metric_used.__name__ == tabular_metrics.auc_metric.__name__:
        if usage == 'sklearn_cv':
            return 'roc_auc_ovo'
        elif usage == 'autogluon':
            #return 'log_loss' # Autogluon crashes when using 'roc_auc' with some datasets usning logloss gives better scores;
                              # We might be able to fix this, but doesn't work out of box.
                              # File bug report? Error happens with dataset robert and fabert
            if multiclass:
                return 'roc_auc_ovo_macro'
            else:
                return 'roc_auc'
        elif usage == 'tabnet':
            return 'logloss' if multiclass else 'auc'
        elif usage == 'autosklearn':
            import autosklearn.classification
            if multiclass:
                return autosklearn.metrics.log_loss # roc_auc only works for binary, use logloss instead
            else:
                return autosklearn.metrics.roc_auc
        elif usage == 'catboost':
            return 'MultiClass' # Effectively LogLoss, ROC not available
        elif usage == 'xgb':
            return 'logloss'
        elif usage == 'lightgbm':
            if multiclass:
                return 'auc'
            else:
                return 'binary'
        return 'roc_auc'
    elif metric_used.__name__ == tabular_metrics.cross_entropy.__name__:
        if usage == 'sklearn_cv':
            return 'neg_log_loss'
        elif usage == 'autogluon':
            return 'log_loss'
        elif usage == 'tabnet':
            return 'logloss'
        elif usage == 'autosklearn':
            import autosklearn.classification
            return autosklearn.metrics.log_loss
        elif usage == 'catboost':
            return 'MultiClass' # Effectively LogLoss
        return 'logloss'
    elif metric_used.__name__ == tabular_metrics.r2_metric.__name__:
        if usage == 'autosklearn':
            import autosklearn.classification
            return autosklearn.metrics.r2
        elif usage == 'sklearn_cv':
            return 'r2' # tabular_metrics.neg_r2
        elif usage == 'autogluon':
            return 'r2'
        elif usage == 'xgb': # XGB cannot directly optimize r2
            return 'rmse'
        elif usage == 'catboost': # Catboost cannot directly optimize r2 ("Can't be used for optimization." - docu)
            return 'RMSE'
        else:
            return 'r2'
    elif metric_used.__name__ == tabular_metrics.root_mean_squared_error_metric.__name__:
        if usage == 'autosklearn':
            import autosklearn.classification
            return autosklearn.metrics.root_mean_squared_error
        elif usage == 'sklearn_cv':
            return 'neg_root_mean_squared_error' # tabular_metrics.neg_r2
        elif usage == 'autogluon':
            return 'rmse'
        elif usage == 'xgb':
            return 'rmse'
        elif usage == 'catboost':
            return 'RMSE'
        else:
            return 'neg_root_mean_squared_error'
    elif metric_used.__name__ == tabular_metrics.mean_absolute_error_metric.__name__:
        if usage == 'autosklearn':
            import autosklearn.classification
            return autosklearn.metrics.mean_absolute_error
        elif usage == 'sklearn_cv':
            return 'neg_mean_absolute_error' # tabular_metrics.neg_r2
        elif usage == 'autogluon':
            return 'mae'
        elif usage == 'xgb':
            return 'mae'
        elif usage == 'catboost':
            return 'MAE'
        else:
            return 'neg_mean_absolute_error'
    else:
        raise Exception('No scoring string found for metric')

def eval_f(params, clf_, x, y, metric_used, start_time, max_time):
    if time.time() - start_time > max_time:
        return np.nan
    scores = cross_val_score(clf_(**params), x, y, cv=CV, scoring=get_scoring_string(metric_used, usage='sklearn_cv'))
    if get_scoring_string(metric_used, usage='sklearn_cv') == 'r2':
        return np.nanmean(scores)
    return -np.nanmean(scores)

def preprocess_impute(x, y, test_x, test_y, impute, one_hot, standardize, cat_features=[]):
    import warnings
    def warn(*args, **kwargs):
        pass

    warnings.warn = warn

    x, y, test_x, test_y = x.cpu().numpy(), y.cpu().long().numpy(), test_x.cpu().numpy(), test_y.cpu().long().numpy()

    if impute:
        imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp_mean.fit(x)
        x, test_x = imp_mean.transform(x), imp_mean.transform(test_x)

    if one_hot:
        def make_pd_from_np(x):
            data = pd.DataFrame(x)
            for c in cat_features:
                data.iloc[:, c] = data.iloc[:, c].astype('int')
            return data
        x, test_x = make_pd_from_np(x),  make_pd_from_np(test_x)
        transformer = ColumnTransformer(transformers=[('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), cat_features)], remainder="passthrough")
        transformer.fit(x)
        x, test_x = transformer.transform(x), transformer.transform(test_x)

    if standardize:
        scaler = MinMaxScaler()
        scaler.fit(x)
        x, test_x = scaler.transform(x), scaler.transform(test_x)

    return x, y, test_x, test_y
import torch
import random
from tqdm import tqdm
def transformer_metric(x, y, test_x, test_y, cat_features, metric_used, max_time=300, device='cpu', N_ensemble_configurations=3, classifier=None):
    from tabpfn.scripts.transformer_prediction_interface import TabPFNClassifier

    if classifier is None:
      classifier = TabPFNClassifier(device=device, N_ensemble_configurations=N_ensemble_configurations)
    classifier.fit(x, y)
    print('Train data shape', x.shape, ' Test data shape', test_x.shape)
    pred = classifier.predict_proba(test_x)

    metric = metric_used(test_y, pred)

    return metric, pred, None

## Auto Gluon
# WARNING: Crashes for some predictors for regression
def autogluon_metric(x, y, test_x, test_y, cat_features, metric_used, max_time=300):
    from autogluon.tabular import TabularPredictor
    x, y, test_x, test_y = preprocess_impute(x, y, test_x, test_y
                                             , one_hot=False
                                             , cat_features=cat_features
                                             , impute=False
                                             , standardize=False)
    train_data = pd.DataFrame(np.concatenate([x, y[:, np.newaxis]], 1))
    test_data = pd.DataFrame(np.concatenate([test_x, test_y[:, np.newaxis]], 1))
    if is_classification(metric_used):
        problem_type = 'multiclass' if len(np.unique(y)) > 2 else 'binary'
    else:
        problem_type = 'regression'
    # AutoGluon automatically infers datatypes, we don't specify the categorical labels
    predictor = TabularPredictor(
        label=train_data.columns[-1],
        eval_metric=get_scoring_string(metric_used, usage='autogluon', multiclass=(len(np.unique(y)) > 2)),
        problem_type=problem_type
        ## seed=int(y[:].sum()) doesn't accept seed
    ).fit(
        train_data=train_data,
        time_limit=max_time,
        presets=['best_quality']
        # The seed is deterministic but varies for each dataset and each split of it
    )

    if is_classification(metric_used):
        pred = predictor.predict_proba(test_data, as_multiclass=True).values
    else:
        pred = predictor.predict(test_data).values

    metric = metric_used(test_y, pred)

    return metric, pred, predictor.fit_summary()

def get_updates_for_regularization_cocktails(
    categorical_indicator: np.ndarray):
    """
    These updates replicate the regularization cocktail paper search space.
    Args:
        categorical_indicator (np.ndarray)
            An array that indicates whether a feature is categorical or not.
        args (Namespace):
            The different updates for the setup of the run, mostly updates
            for the different regularization ingredients.
    Returns:
    ________
        pipeline_update, search_space_updates, include_updates (Tuple[dict, HyperparameterSearchSpaceUpdates, dict]):
            The pipeline updates like number of epochs, budget, seed etc.
            The search space updates like setting different hps to different values or ranges.
            Lastly include updates, which can be used to include different features.
    """
    from autoPyTorch.utils.hyperparameter_search_space_update import HyperparameterSearchSpaceUpdates
    import argparse

    augmentation_names_to_trainers = {
        'mixup': 'MixUpTrainer',
        'cutout': 'RowCutOutTrainer',
        'cutmix': 'RowCutMixTrainer',
        'standard': 'StandardTrainer',
        'adversarial': 'AdversarialTrainer',
    }

    include_updates = dict()
    include_updates['network_embedding'] = ['NoEmbedding']
    include_updates['network_init'] = ['NoInit']

    has_cat_features = any(categorical_indicator)
    has_numerical_features = not all(categorical_indicator)

    def str2bool(v):
        if isinstance(v, bool):
            return [v, ]
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return [True, ]
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return [False, ]
        elif v.lower() == 'conditional':
            return [True, False]
        else:
            raise ValueError('No valid value given.')
    search_space_updates = HyperparameterSearchSpaceUpdates()

    # architecture head
    search_space_updates.append(
        node_name='network_head',
        hyperparameter='__choice__',
        value_range=['no_head'],
        default_value='no_head',
    )
    search_space_updates.append(
        node_name='network_head',
        hyperparameter='no_head:activation',
        value_range=['relu'],
        default_value='relu',
    )

    # backbone architecture
    search_space_updates.append(
        node_name='network_backbone',
        hyperparameter='__choice__',
        value_range=['ShapedResNetBackbone'],
        default_value='ShapedResNetBackbone',
    )
    search_space_updates.append(
        node_name='network_backbone',
        hyperparameter='ShapedResNetBackbone:resnet_shape',
        value_range=['brick'],
        default_value='brick',
    )
    search_space_updates.append(
        node_name='network_backbone',
        hyperparameter='ShapedResNetBackbone:num_groups',
        value_range=[2],
        default_value=2,
    )
    search_space_updates.append(
        node_name='network_backbone',
        hyperparameter='ShapedResNetBackbone:blocks_per_group',
        value_range=[2],
        default_value=2,
    )
    search_space_updates.append(
        node_name='network_backbone',
        hyperparameter='ShapedResNetBackbone:output_dim',
        value_range=[512],
        default_value=512,
    )
    search_space_updates.append(
        node_name='network_backbone',
        hyperparameter='ShapedResNetBackbone:max_units',
        value_range=[512],
        default_value=512,
    )
    search_space_updates.append(
        node_name='network_backbone',
        hyperparameter='ShapedResNetBackbone:activation',
        value_range=['relu'],
        default_value='relu',
    )
    search_space_updates.append(
        node_name='network_backbone',
        hyperparameter='ShapedResNetBackbone:shake_shake_update_func',
        value_range=['even-even'],
        default_value='even-even',
    )

    # training updates
    search_space_updates.append(
        node_name='lr_scheduler',
        hyperparameter='__choice__',
        value_range=['CosineAnnealingWarmRestarts'],
        default_value='CosineAnnealingWarmRestarts',
    )
    search_space_updates.append(
        node_name='lr_scheduler',
        hyperparameter='CosineAnnealingWarmRestarts:n_restarts',
        value_range=[3],
        default_value=3,
    )
    search_space_updates.append(
        node_name='optimizer',
        hyperparameter='__choice__',
        value_range=['AdamWOptimizer'],
        default_value='AdamWOptimizer',
    )
    search_space_updates.append(
        node_name='optimizer',
        hyperparameter='AdamWOptimizer:lr',
        value_range=[1e-3],
        default_value=1e-3,
    )
    search_space_updates.append(
        node_name='data_loader',
        hyperparameter='batch_size',
        value_range=[128],
        default_value=128,
    )

    # preprocessing
    search_space_updates.append(
        node_name='feature_preprocessor',
        hyperparameter='__choice__',
        value_range=['NoFeaturePreprocessor'],
        default_value='NoFeaturePreprocessor',
    )

    if has_numerical_features:
        print('has numerical features')
        search_space_updates.append(
            node_name='imputer',
            hyperparameter='numerical_strategy',
            value_range=['median'],
            default_value='median',
        )
        search_space_updates.append(
            node_name='scaler',
            hyperparameter='__choice__',
            value_range=['StandardScaler'],
            default_value='StandardScaler',
        )

    if has_cat_features:
        print('has cat features')
        search_space_updates.append(
            node_name='imputer',
            hyperparameter='categorical_strategy',
            value_range=['constant_!missing!'],
            default_value='constant_!missing!',
        )
        search_space_updates.append(
            node_name='encoder',
            hyperparameter='__choice__',
            value_range=['OneHotEncoder'],
            default_value='OneHotEncoder',
        )

    search_space_updates.append(
        node_name='optimizer',
        hyperparameter='AdamWOptimizer:beta1',
        value_range=[0.9],
        default_value=0.9,
    )
    search_space_updates.append(
        node_name='optimizer',
        hyperparameter='AdamWOptimizer:beta2',
        value_range=[0.999],
        default_value=0.999,
    )


    parser = argparse.ArgumentParser(
        description='Run AutoPyTorch on a benchmark.',
    )
    # experiment setup arguments
    parser.add_argument(
        '--task_id',
        type=int,
        default=233088,
    )
    parser.add_argument(
        '--wall_time',
        type=int,
        default=9000,
    )
    parser.add_argument(
        '--func_eval_time',
        type=int,
        default=1000,
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=105,
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=11,
    )
    parser.add_argument(
        '--tmp_dir',
        type=str,
        default='./runs/autoPyTorch_cocktails',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./runs/autoPyTorch_cocktails',
    )
    parser.add_argument(
        '--nr_workers',
        type=int,
        default=1,
    )
    parser.add_argument(
        '--nr_threads',
        type=int,
        default=1,
    )
    parser.add_argument(
        '--cash_cocktail',
        help='If the regularization cocktail should be used.',
        type=bool,
        default=False,
    )

    # regularization ingredient arguments
    parser.add_argument(
        '--use_swa',
        help='If stochastic weight averaging should be used.',
        type=str2bool,
        nargs='?',
        const=[True],
        default=[False],
    )
    parser.add_argument(
        '--use_se',
        help='If snapshot ensembling should be used.',
        type=str2bool,
        nargs='?',
        const=[True],
        default=[False],
    )
    parser.add_argument(
        '--use_lookahead',
        help='If the lookahead optimizing technique should be used.',
        type=str2bool,
        nargs='?',
        const=[True],
        default=[False],
    )
    parser.add_argument(
        '--use_weight_decay',
        help='If weight decay regularization should be used.',
        type=str2bool,
        nargs='?',
        const=[True],
        default=[False],
    )
    parser.add_argument(
        '--use_batch_normalization',
        help='If batch normalization regularization should be used.',
        type=str2bool,
        nargs='?',
        const=[True],
        default=[False],
    )
    parser.add_argument(
        '--use_skip_connection',
        help='If skip connections should be used. '
             'Turns the network into a residual network.',
        type=str2bool,
        nargs='?',
        const=[True],
        default=[False],
    )
    parser.add_argument(
        '--use_dropout',
        help='If dropout regularization should be used.',
        type=str2bool,
        nargs='?',
        const=[True],
        default=[False],
    )
    parser.add_argument(
        '--mb_choice',
        help='Multibranch network regularization. '
             'Only active when skip_connection is active.',
        type=str,
        choices=['none', 'shake-shake', 'shake-drop'],
        default='none',
    )
    parser.add_argument(
        '--augmentation',
        help='If methods that augment examples should be used',
        type=str,
        choices=['mixup', 'cutout', 'cutmix', 'standard', 'adversarial'],
        default='standard',
    )

    args = parser.parse_args([]) # just get default values



    # if the cash formulation of the cocktail is not activated,
    # otherwise the methods activation will be chosen by the SMBO optimizer.


    # No early stopping and train on gpu
    pipeline_update = {
        'early_stopping': -1,
        'min_epochs': args.epochs,
        'epochs': args.epochs,
        "device": 'cpu',
    }

    return pipeline_update, search_space_updates, include_updates

def get_smac_object(
    scenario_dict,
    seed: int,
    ta,
    ta_kwargs,
    n_jobs: int,
    initial_budget: int,
    max_budget: int,
    dask_client,
):
    """
    This function returns an SMAC object that is gonna be used as
    optimizer of pipelines.
    Args:
        scenario_dict (typing.Dict[str, typing.Any]): constrain on how to run
            the jobs.
        seed (int): to make the job deterministic.
        ta (typing.Callable): the function to be intensified by smac.
        ta_kwargs (typing.Dict[str, typing.Any]): Arguments to the above ta.
        n_jobs (int): Amount of cores to use for this task.
        initial_budget (int):
            The initial budget for a configuration.
        max_budget (int):
            The maximal budget for a configuration.
        dask_client (dask.distributed.Client): User provided scheduler.
    Returns:
        (SMAC4AC): sequential model algorithm configuration object
    """
    from smac.intensification.simple_intensifier import SimpleIntensifier
    from smac.runhistory.runhistory2epm import RunHistory2EPM4LogCost
    from smac.scenario.scenario import Scenario
    from smac.facade.smac_ac_facade import SMAC4AC
    # multi-fidelity is disabled, that is why initial_budget and max_budget
    # are not used.
    rh2EPM = RunHistory2EPM4LogCost

    return SMAC4AC(
        scenario=Scenario(scenario_dict),
        rng=seed,
        runhistory2epm=rh2EPM,
        tae_runner=ta,
        tae_runner_kwargs=ta_kwargs,
        initial_configurations=None,
        run_id=seed,
        intensifier=SimpleIntensifier,
        dask_client=dask_client,
        n_jobs=n_jobs,
    )


def get_incumbent_results(
    run_history_file: str,
    search_space
):
    """
    Get the incumbent configuration and performance from the previous run HPO
    search with AutoPytorch.
    Args:
        run_history_file (str):
            The path where the AutoPyTorch search data is located.
        search_space (ConfigSpace.ConfigurationSpace):
            The ConfigurationSpace that was previously used for the HPO
            search space.
    Returns:
        config, incumbent_run_value (Tuple[ConfigSpace.Configuration, float]):
            The incumbent configuration found from HPO search and the validation
            performance it achieved.
    """
    from smac.runhistory.runhistory import RunHistory
    run_history = RunHistory()
    run_history.load_json(
        run_history_file,
        search_space,
    )

    run_history_data = run_history.data
    sorted_runvalue_by_cost = sorted(run_history_data.items(), key=lambda item: item[1].cost)
    incumbent_run_key, incumbent_run_value = sorted_runvalue_by_cost[0]
    config = run_history.ids_config[incumbent_run_key.config_id]
    return config, incumbent_run_value


def well_tuned_simple_nets_metric(X_train, y_train, X_test, y_test, categorical_indicator, metric_used, max_time=300, nr_workers=1):
    """Install:
    git clone https://github.com/automl/Auto-PyTorch.git
    cd Auto-PyTorch
    git checkout regularization_cocktails
    From the page, not needed for me at least: conda install gxx_linux-64 gcc_linux-64 swig
    conda create --clone CONDANAME --name CLONENAME
    conda activate CLONENAME
    pip install -r requirements.txt (I checked looks like nothing should break functionality of our project not sure about baselines, thus a copied env is likely good :))
    pip install -e .
    """
    #os.environ.get('SLURM_JOBID', '')
    categorical_indicator = np.array([i in categorical_indicator for i in range(X_train.shape[1])])
    with tempfile.TemporaryDirectory(prefix=f"{len(X_train)}_{len(X_test)}_{max_time}") as temp_dir:
        from autoPyTorch.api.tabular_classification import TabularClassificationTask
        from autoPyTorch.datasets.resampling_strategy import HoldoutValTypes, NoResamplingStrategyTypes
        from autoPyTorch.data.tabular_validator import TabularInputValidator
        from autoPyTorch.datasets.tabular_dataset import TabularDataset
        from autoPyTorch import metrics
        # append random folder to temp_dir to avoid collisions
        rand_int = str(random.randint(1,1000))
        temp_dir = os.path.join(temp_dir, 'temp_'+rand_int)
        out_dir = os.path.join(temp_dir, 'out_'+rand_int)

        start_time = time.time()

        X_train, y_train, X_test, y_test = X_train.cpu().numpy(), y_train.cpu().long().numpy(), X_test.cpu().numpy(), y_test.cpu().long().numpy()

        def safe_int(x):
            assert np.all(x.astype('int64') == x) or np.any(x != x), np.unique(x) # second condition for ignoring nans
            return pd.Series(x, dtype='category')

        X_train = pd.DataFrame({i: safe_int(X_train[:,i]) if c else X_train[:,i] for i, c in enumerate(categorical_indicator)})
        X_test = pd.DataFrame({i: safe_int(X_test[:,i]) if c else X_test[:,i] for i, c in enumerate(categorical_indicator)})


        if isinstance(y_train[1], bool):
            y_train = y_train.astype('bool')
        if isinstance(y_test[1], bool):
            y_test = y_test.astype('bool')

        number_of_configurations_limit = 840 # hard coded in the paper
        epochs = 105
        func_eval_time = min(1000, max_time/2)
        seed = int(y_train[:].sum())


        resampling_strategy_args = {
            'val_share': len(y_test)/(len(y_test)+len(y_train)),
        }

        pipeline_update, search_space_updates, include_updates = get_updates_for_regularization_cocktails(
            categorical_indicator,
        )
        print(search_space_updates)



        ############################################################################
        # Build and fit a classifier
        # ==========================
        # if we use HPO, we can use multiple workers in parallel
        if number_of_configurations_limit == 0:
            nr_workers = 1

        api = TabularClassificationTask(
            temporary_directory=temp_dir,
            output_directory=out_dir,
            delete_tmp_folder_after_terminate=False,
            delete_output_folder_after_terminate=False,
            resampling_strategy=HoldoutValTypes.stratified_holdout_validation,
            resampling_strategy_args=resampling_strategy_args,
            ensemble_size=1,
            ensemble_nbest=1,
            max_models_on_disc=10,
            include_components=include_updates,
            search_space_updates=search_space_updates,
            seed=seed,
            n_jobs=nr_workers,
            n_threads=1,
        )

        api.set_pipeline_config(**pipeline_update)
        ############################################################################
        # Search for the best hp configuration
        # ====================================
        # We search for the best hp configuration only in the case of a cocktail ingredient
        # that has hyperparameters.
        print(X_train, X_test)
        print('temp_dir',temp_dir)
        print(max_time, min(func_eval_time, max_time, number_of_configurations_limit))

        if number_of_configurations_limit != 0:
            api.search(
                X_train=X_train.copy(),
                y_train=y_train.copy(),
                X_test=X_test.copy(),
                y_test=y_test.copy(),
                optimize_metric='balanced_accuracy',
                total_walltime_limit=max_time,
                memory_limit=12000,
                func_eval_time_limit_secs=min(func_eval_time, max_time),
                enable_traditional_pipeline=False,
                get_smac_object_callback=get_smac_object,
                smac_scenario_args={
                    'runcount_limit': number_of_configurations_limit,
                },
            )

        ############################################################################
        # Refit on the best hp configuration
        # ==================================
        input_validator = TabularInputValidator(
            is_classification=True,
        )
        input_validator.fit(
            X_train=X_train.copy(),
            y_train=y_train.copy(),
            X_test=X_test.copy(),
            y_test=y_test.copy(),
        )

        dataset = TabularDataset(
            X=X_train,
            Y=y_train,
            X_test=X_test,
            Y_test=y_test,
            seed=seed,
            validator=input_validator,
            resampling_strategy=NoResamplingStrategyTypes.no_resampling,
        )
        dataset.is_small_preprocess = False
        print(f"Fitting pipeline with {epochs} epochs")

        search_space = api.get_search_space(dataset)
        # only when we perform hpo will there be an incumbent configuration
        # otherwise take a default configuration.
        if number_of_configurations_limit != 0:
            configuration, incumbent_run_value = get_incumbent_results(
                os.path.join(
                    temp_dir,
                    'smac3-output',
                    'run_{}'.format(seed),
                    'runhistory.json'),
                search_space,
            )
            print(f"Incumbent configuration: {configuration}")
            print(f"Incumbent trajectory: {api.trajectory}")
        else:
            # default configuration
            configuration = search_space.get_default_configuration()
            print(f"Default configuration: {configuration}")

        fitted_pipeline, run_info, run_value, dataset = api.fit_pipeline(
            configuration=configuration,
            budget_type='epochs',
            budget=epochs,
            dataset=dataset,
            run_time_limit_secs=func_eval_time,
            eval_metric='balanced_accuracy',
            memory_limit=12000,
        )

        X_train = dataset.train_tensors[0]
        y_train = dataset.train_tensors[1]
        X_test = dataset.test_tensors[0]
        y_test = dataset.test_tensors[1]

        if fitted_pipeline is None:
            api.get_incumbent_config


        train_predictions = fitted_pipeline.predict(X_train)
        test_predictions = fitted_pipeline.predict(X_test)

        metric = metric_used(y_test, test_predictions.squeeze())
        duration = time.time() - start_time

        print(f'Time taken: {duration} for {metric} metric')
        print(test_predictions[:10])
        return metric, test_predictions, None



## AUTO Sklearn
def autosklearn_metric(x, y, test_x, test_y, cat_features, metric_used, max_time=300):
    import autosklearn.classification
    return autosklearn2_metric(x, y, test_x, test_y, cat_features, metric_used, max_time=max_time, version=1)

def autosklearn2_metric(x, y, test_x, test_y, cat_features, metric_used, max_time=300, version=2):
    from autosklearn.experimental.askl2 import AutoSklearn2Classifier
    from autosklearn.classification import AutoSklearnClassifier
    from autosklearn.regression import AutoSklearnRegressor
    x, y, test_x, test_y = preprocess_impute(x, y, test_x, test_y
                                             , one_hot=False
                                             , cat_features=cat_features
                                             , impute=False
                                             , standardize=False)

    def make_pd_from_np(x):
        data = pd.DataFrame(x)
        for c in cat_features:
            data.iloc[:, c] = data.iloc[:, c].astype('category')
        return data

    x = make_pd_from_np(x)
    test_x = make_pd_from_np(test_x)

    if is_classification(metric_used):
        clf_ = AutoSklearn2Classifier if version == 2 else AutoSklearnClassifier
    else:
        if version == 2:
            raise Exception("AutoSklearn 2 doesn't do regression.")
        clf_ = AutoSklearnRegressor
    clf = clf_(time_left_for_this_task=max_time,
                                                           memory_limit=4000,
                                                           n_jobs=MULTITHREAD,
               seed=int(y[:].sum()),
        # The seed is deterministic but varies for each dataset and each split of it
               metric=get_scoring_string(metric_used, usage='autosklearn', multiclass=len(np.unique(y)) > 2))

    # fit model to data
    clf.fit(x, y)

    if is_classification(metric_used):
        pred = clf.predict_proba(test_x)
    else:
        pred = clf.predict(test_x)
    metric = metric_used(test_y, pred)

    return metric, pred, None

param_grid_hyperopt['ridge'] = {
    'max_iter': hp.randint('max_iter', 50, 500)
    , 'fit_intercept': hp.choice('fit_intercept', [True, False])
    , 'alpha': hp.loguniform('alpha', -5, math.log(5.0))}  # 'normalize': [False],

def ridge_metric(x, y, test_x, test_y, cat_features, metric_used, max_time=300):
    if is_classification(metric_used):
        raise Exception("Ridge is only applicable to pointwise Regression.")

    x, y, test_x, test_y = preprocess_impute(x, y, test_x, test_y
                                             , one_hot=True, impute=True, standardize=True
                                             , cat_features=cat_features)
    def clf_(**params):
        return Ridge(tol=1e-4, **params)

    start_time = time.time()

    def stop(trial):
        return time.time() - start_time > max_time, []

    best = fmin(
        fn=lambda params: eval_f(params, clf_, x, y, metric_used, start_time, max_time),
        space=param_grid_hyperopt['ridge'],
        algo=rand.suggest,
        rstate=np.random.RandomState(int(y[:].sum()) % 10000),
        early_stop_fn=stop,
        # The seed is deterministic but varies for each dataset and each split of it
        max_evals=10000)
    best = space_eval(param_grid_hyperopt['ridge'], best)

    clf = clf_(**best)
    clf.fit(x, y)

    pred = clf.predict(test_x)
    metric = metric_used(test_y, pred)

    return metric, pred, best

def lightautoml_metric(x, y, test_x, test_y, cat_features, metric_used, max_time=300):
    from lightautoml.automl.presets.tabular_presets import TabularAutoML, TabularUtilizedAutoML
    from lightautoml.tasks import Task
    
    x, y, test_x, test_y = preprocess_impute(x, y, test_x, test_y
                                             , one_hot=False, impute=False, standardize=False
                                             , cat_features=cat_features)

    roles = {'target': str(x.shape[-1])}
    task = Task('multiclass', metric = lambda x, y : metric_used(x, y, numpy=True))
    automl = TabularUtilizedAutoML(task=task,
                           timeout=max_time,
                           cpu_limit=4,  # Optimal for Kaggle kernels
                           general_params={'use_algos': [['linear_l2',
                                                          'lgb', 'lgb_tuned']]})

    tr_data = np.concatenate([x, np.expand_dims(y, -1)], -1)
    tr_data = pd.DataFrame(tr_data, columns=[str(k) for k in range(0, x.shape[-1] + 1)])
    oof_pred = automl.fit_predict(tr_data, roles=roles)
    te_data = pd.DataFrame(test_x, columns=[str(k) for k in range(0, x.shape[-1])])

    probabilities = automl.predict(te_data).data
    probabilities_mapped = probabilities.copy()

    class_map = automl.outer_pipes[0].ml_algos[0].models[0][0].reader.class_mapping
    if class_map:
        column_to_class = {col: class_ for class_, col in class_map.items()}
        for i in range(0, len(column_to_class)):
            probabilities_mapped[:, int(column_to_class[int(i)])] = probabilities[:, int(i)]

    metric = metric_used(test_y, probabilities_mapped)

    return metric, probabilities_mapped, None

param_grid_hyperopt['lightgbm'] = {
    'num_leaves': hp.randint('num_leaves', 5, 50)
    , 'max_depth': hp.randint('max_depth', 3, 20)
    , 'learning_rate': hp.loguniform('learning_rate', -3, math.log(1.0))
    , 'n_estimators': hp.randint('n_estimators', 50, 2000)
    #, 'feature_fraction': 0.8,
    #, 'subsample': 0.2
    , 'min_child_weight': hp.choice('min_child_weight', [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4])
    , 'subsample': hp.uniform('subsample', 0.2, 0.8)
    , 'colsample_bytree': hp.uniform('colsample_bytree', 0.2, 0.8)
    , 'reg_alpha': hp.choice('reg_alpha', [0, 1e-1, 1, 2, 5, 7, 10, 50, 100])
    , 'reg_lambda': hp.choice('reg_lambda', [0, 1e-1, 1, 5, 10, 20, 50, 100])
}  # 'normalize': [False],

from lightgbm import LGBMClassifier

def lightgbm_metric(x, y, test_x, test_y, cat_features, metric_used, max_time=300):
    x, y, test_x, test_y = preprocess_impute(x, y, test_x, test_y
                                             , one_hot=False, impute=False, standardize=False
                                             , cat_features=cat_features)

    def clf_(**params):
        return LGBMClassifier(categorical_feature=cat_features, use_missing=True
                              , objective=get_scoring_string(metric_used, usage='lightgbm', multiclass=len(np.unique(y)) > 2), **params)

    start_time = time.time()

    def stop(trial):
        return time.time() - start_time > max_time, []

    best = fmin(
        fn=lambda params: eval_f(params, clf_, x, y, metric_used, start_time, max_time),
        space=param_grid_hyperopt['lightgbm'],
        algo=rand.suggest,
        rstate=np.random.RandomState(int(y[:].sum()) % 10000),
        early_stop_fn=stop,
        # The seed is deterministic but varies for each dataset and each split of it
        max_evals=10000)
    best = space_eval(param_grid_hyperopt['lightgbm'], best)

    clf = clf_(**best)
    clf.fit(x, y)

    pred = clf.predict_proba(test_x)
    metric = metric_used(test_y, pred)

    return metric, pred, best

param_grid_hyperopt['logistic'] = {
    'penalty': hp.choice('penalty', ['l1', 'l2', 'none'])
    , 'max_iter': hp.randint('max_iter', 50, 500)
    , 'fit_intercept': hp.choice('fit_intercept', [True, False])
    , 'C': hp.loguniform('C', -5, math.log(5.0))}  # 'normalize': [False],


def logistic_metric(x, y, test_x, test_y, cat_features, metric_used, max_time=300):
    x, y, test_x, test_y = preprocess_impute(x, y, test_x, test_y
                                             , one_hot=True, impute=True, standardize=True
                                             , cat_features=cat_features)

    def clf_(**params):
        return LogisticRegression(solver='saga', tol=1e-4, n_jobs=MULTITHREAD, **params)

    start_time = time.time()

    def stop(trial):
        return time.time() - start_time > max_time, []

    best = fmin(
        fn=lambda params: eval_f(params, clf_, x, y, metric_used, start_time, max_time),
        space=param_grid_hyperopt['logistic'],
        algo=rand.suggest,
        rstate=np.random.RandomState(int(y[:].sum()) % 10000),
        early_stop_fn=stop,
        # The seed is deterministic but varies for each dataset and each split of it
        max_evals=1000)
    best = space_eval(param_grid_hyperopt['logistic'], best)

    clf = clf_(**best)
    clf.fit(x, y)

    pred = clf.predict_proba(test_x)
    metric = metric_used(test_y, pred)

    return metric, pred, best


## Random Forest
# Search space from
# https://www.kaggle.com/code/emanueleamcappella/random-forest-hyperparameters-tuning/notebook
param_grid_hyperopt['random_forest'] = {'n_estimators': hp.randint('n_estimators', 20, 200),
               'max_features': hp.choice('max_features', ['auto', 'sqrt']),
               'max_depth': hp.randint('max_depth', 1, 45),
               'min_samples_split': hp.choice('min_samples_split', [5, 10])}
def random_forest_metric(x, y, test_x, test_y, cat_features, metric_used, max_time=300):
    from sklearn.ensemble import RandomForestClassifier

    x, y, test_x, test_y = preprocess_impute(x, y, test_x, test_y,
                                             one_hot=False, impute=True, standardize=False,
                                             cat_features=cat_features)

    def clf_(**params):
        if is_classification(metric_used):
            return RandomForestClassifier(n_jobs=MULTITHREAD, **params)
        return RandomForestClassifier(n_jobs=MULTITHREAD, **params)

    start_time = time.time()

    def stop(trial):
        return time.time() - start_time > max_time, []

    best = fmin(
        fn=lambda params: eval_f(params, clf_, x, y, metric_used, start_time, max_time),
        space=param_grid_hyperopt['random_forest'],
        algo=rand.suggest,
        rstate=np.random.RandomState(int(y[:].sum()) % 10000),
        early_stop_fn=stop,
        # The seed is deterministic but varies for each dataset and each split of it
        max_evals=10000)
    best = space_eval(param_grid_hyperopt['random_forest'], best)

    clf = clf_(**best)
    clf.fit(x, y)

    if is_classification(metric_used):
        pred = clf.predict_proba(test_x)
    else:
        pred = clf.predict(test_x)
    metric = metric_used(test_y, pred)

    return metric, pred, best

## KNN
param_grid_hyperopt['knn'] = {'n_neighbors': hp.randint('n_neighbors', 1,16)
                              }
def knn_metric(x, y, test_x, test_y, cat_features, metric_used, max_time=300):
    x, y, test_x, test_y = preprocess_impute(x, y, test_x, test_y,
                                             one_hot=True, impute=True, standardize=True,
                                             cat_features=cat_features)

    def clf_(**params):
        if is_classification(metric_used):
            return neighbors.KNeighborsClassifier(n_jobs=MULTITHREAD, **params)
        return neighbors.KNeighborsRegressor(n_jobs=MULTITHREAD, **params)

    start_time = time.time()

    def stop(trial):
        return time.time() - start_time > max_time, []

    best = fmin(
        fn=lambda params: eval_f(params, clf_, x, y, metric_used, start_time, max_time),
        space=param_grid_hyperopt['knn'],
        algo=rand.suggest,
        rstate=np.random.RandomState(int(y[:].sum()) % 10000),
        early_stop_fn=stop,
        # The seed is deterministic but varies for each dataset and each split of it
        max_evals=200)
    best = space_eval(param_grid_hyperopt['knn'], best)

    clf = clf_(**best)
    clf.fit(x, y)

    if is_classification(metric_used):
        pred = clf.predict_proba(test_x)
    else:
        pred = clf.predict(test_x)
    metric = metric_used(test_y, pred)

    return metric, pred, best

## GP
param_grid_hyperopt['gp'] = {
    'params_y_scale': hp.loguniform('params_y_scale', math.log(0.05), math.log(5.0)),
    'params_length_scale': hp.loguniform('params_length_scale', math.log(0.1), math.log(1.0))
}
def gp_metric(x, y, test_x, test_y, cat_features, metric_used, max_time=300):
    x, y, test_x, test_y = preprocess_impute(x, y, test_x, test_y,
                                             one_hot=True, impute=True, standardize=True,
                                             cat_features=cat_features)

    def clf_(params_y_scale,params_length_scale, **params):
        if is_classification(metric_used):
            return GaussianProcessClassifier(kernel= params_y_scale * RBF(params_length_scale), **params)
        else:
            return GaussianProcessRegressor(kernel= params_y_scale * RBF(params_length_scale), **params)

    start_time = time.time()
    def stop(trial):
        return time.time() - start_time > max_time, []


    best = fmin(
        fn=lambda params: eval_f(params, clf_, x, y, metric_used, start_time, max_time),
        space=param_grid_hyperopt['gp'],
        algo=rand.suggest,
        rstate=np.random.RandomState(int(y[:].sum()) % 10000),
        early_stop_fn=stop,
        # The seed is deterministic but varies for each dataset and each split of it
        max_evals=1000)
    best = space_eval(param_grid_hyperopt['gp'], best)

    clf = clf_(**best)
    clf.fit(x, y)

    if is_classification(metric_used):
        pred = clf.predict_proba(test_x)
    else:
        pred = clf.predict(test_x)
    metric = metric_used(test_y, pred)

    return metric, pred, best

## Tabnet
# https://github.com/dreamquark-ai/tabnet
#param_grid['tabnet'] = {'n_d': [2, 4], 'n_steps': [2,4,6], 'gamma': [1.3], 'optimizer_params': [{'lr': 2e-2}, {'lr': 2e-1}]}

# Hyperparameter space from dreamquarks implementation recommendations
param_grid_hyperopt['tabnet'] = {
    'n_d': hp.randint('n_d', 8, 64),
    'n_steps': hp.randint('n_steps', 3, 10),
    'max_epochs': hp.randint('max_epochs', 50, 200),
    'gamma': hp.uniform('relax', 1.0, 2.0),
    'momentum': hp.uniform('momentum', 0.01, 0.4),
}

def tabnet_metric(x, y, test_x, test_y, cat_features, metric_used, max_time=300):
    from pytorch_tabnet.tab_model import TabNetClassifier
    # TabNet inputs raw tabular data without any preprocessing and is trained using gradient descent-based optimisation.
    # However Tabnet cannot handle nans so we impute with mean

    x, y, test_x, test_y = preprocess_impute(x, y, test_x, test_y, impute=True, one_hot=False, standardize=False)

    def clf_(**params):
        return TabNetClassifier(cat_idxs=cat_features, verbose=True, n_a=params['n_d'], seed=int(y[:].sum()), **params)

    def tabnet_eval_f(params, clf_, x, y, metric_used, start_time, max_time):
        if time.time() - start_time > max_time:
            return np.nan

        kf = KFold(n_splits=min(CV, x.shape[0] // 2), random_state=None, shuffle=True)
        metrics = []

        params = {**params}
        max_epochs = params['max_epochs']
        del params['max_epochs']

        for train_index, test_index in kf.split(x):
            X_train, X_valid, y_train, y_valid = x[train_index], x[test_index], y[train_index], y[test_index]

            clf = clf_(**params)

            clf.fit(
                X_train, y_train,
                # eval_metric=[get_scoring_string(metric_used, multiclass=len(np.unique(y_train)) > 2, usage='tabnet')],
                # eval_set=[(X_valid, y_valid)],
                # patience=15,
                max_epochs=max_epochs
            )
            metrics += [metric_used(y_valid, clf.predict_proba(X_valid))]

        return -np.nanmean(np.array(metrics))

    start_time = time.time()
    def stop(trial):
        return time.time() - start_time > max_time, []

    best = fmin(
        fn=lambda params: tabnet_eval_f(params, clf_, x, y, metric_used, start_time, max_time),
        space=param_grid_hyperopt['tabnet'],
        algo=rand.suggest,
        rstate=np.random.RandomState(int(y[:].sum()) % 10000),
        early_stop_fn=stop,
        max_evals=1000)
    best = space_eval(param_grid_hyperopt['tabnet'], best)
    max_epochs = best['max_epochs']
    del best['max_epochs']

    clf = clf_(**best)
    clf.fit(x, y, max_epochs=max_epochs) # , max_epochs=mean_best_epochs[best_idx]

    pred = clf.predict_proba(test_x)
    metric = metric_used(test_y, pred)

    return metric, pred, best

    return metric, pred, params_used[best_idx]


# Catboost
# Hyperparameter space: https://arxiv.org/pdf/2106.03253.pdf

param_grid_hyperopt['catboost'] = {
    'learning_rate': hp.loguniform('learning_rate', math.log(math.pow(math.e, -5)), math.log(1)),
    'random_strength': hp.randint('random_strength', 1, 20),
    'l2_leaf_reg': hp.loguniform('l2_leaf_reg', math.log(1), math.log(10)),
    'bagging_temperature': hp.uniform('bagging_temperature', 0., 1),
    'leaf_estimation_iterations': hp.randint('leaf_estimation_iterations', 1, 20),
    'iterations': hp.randint('iterations', 100, 4000), # This is smaller than in paper, 4000 leads to ram overusage
}

def catboost_metric(x, y, test_x, test_y, cat_features, metric_used, max_time=300):
    x, y, test_x, test_y = preprocess_impute(x, y, test_x, test_y
                                             , one_hot=False
                                             , cat_features=cat_features
                                             , impute=False
                                             , standardize=False)

    # Nans in categorical features must be encoded as separate class
    x[:, cat_features], test_x[:, cat_features] = np.nan_to_num(x[:, cat_features], -1), np.nan_to_num(
        test_x[:, cat_features], -1)

    def make_pd_from_np(x):
        data = pd.DataFrame(x)
        for c in cat_features:
            data.iloc[:, c] = data.iloc[:, c].astype('int')
        return data

    x = make_pd_from_np(x)
    test_x = make_pd_from_np(test_x)

    def clf_(**params):
        if is_classification(metric_used):
            return CatBoostClassifier(
                                   loss_function=get_scoring_string(metric_used, usage='catboost'),
                                   thread_count = MULTITHREAD,
                                   used_ram_limit='4gb',
                random_seed=int(y[:].sum()),
                                   logging_level='Silent',
                                    cat_features=cat_features,
                                      **params)
        else:
            return CatBoostRegressor(
                loss_function=get_scoring_string(metric_used, usage='catboost'),
                thread_count=MULTITHREAD,
                used_ram_limit='4gb',
                random_seed=int(y[:].sum()),
                logging_level='Silent',
                cat_features=cat_features,
                **params)

    start_time = time.time()
    def stop(trial):
        return time.time() - start_time > max_time, []

    best = fmin(
        fn=lambda params: eval_f(params, clf_, x, y, metric_used, start_time, max_time),
        space=param_grid_hyperopt['catboost'],
        algo=rand.suggest,
        rstate=np.random.RandomState(int(y[:].sum()) % 10000),
        early_stop_fn=stop,
        # The seed is deterministic but varies for each dataset and each split of it
        max_evals=1000)
    best = space_eval(param_grid_hyperopt['catboost'], best)

    clf = clf_(**best)
    clf.fit(x, y)
    if is_classification(metric_used):
        pred = clf.predict_proba(test_x)
    else:
        pred = clf.predict(test_x)
    metric = metric_used(test_y, pred)

    return metric, pred, best


# XGBoost
# Hyperparameter space: https://arxiv.org/pdf/2106.03253.pdf
param_grid_hyperopt['xgb'] = {
    'learning_rate': hp.loguniform('learning_rate', -7, math.log(1)),
    'max_depth': hp.randint('max_depth', 1, 10),
    'subsample': hp.uniform('subsample', 0.2, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.2, 1),
    'colsample_bylevel': hp.uniform('colsample_bylevel', 0.2, 1),
    'min_child_weight': hp.loguniform('min_child_weight', -16, 5),
    'alpha': hp.loguniform('alpha', -16, 2),
    'lambda': hp.loguniform('lambda', -16, 2),
    'gamma': hp.loguniform('gamma', -16, 2),
    'n_estimators': hp.randint('n_estimators', 100, 4000), # This is smaller than in paper
}

def xgb_metric(x, y, test_x, test_y, cat_features, metric_used, max_time=300):
    import xgboost as xgb
    # XGB Documentation:
    # XGB handles categorical data appropriately without using One Hot Encoding, categorical features are experimetal
    # XGB handles missing values appropriately without imputation

    x, y, test_x, test_y = preprocess_impute(x, y, test_x, test_y
                                             , one_hot=False
                                             , cat_features=cat_features
                                             , impute=False
                                             , standardize=False)

    def clf_(**params):
        if is_classification(metric_used):
            return xgb.XGBClassifier(use_label_encoder=False
                                     , nthread=1
                                     , **params
                                     , eval_metric=get_scoring_string(metric_used, usage='xgb') # AUC not implemented
            )
        else:
            return xgb.XGBRegressor(use_label_encoder=False
                                     , nthread=1
                                     , **params
                                     , eval_metric=get_scoring_string(metric_used, usage='xgb')  # AUC not implemented
                                     )

    start_time = time.time()
    def stop(trial):
        return time.time() - start_time > max_time, []

    best = fmin(
        fn=lambda params: eval_f(params, clf_, x, y, metric_used, start_time, max_time),
        space=param_grid_hyperopt['xgb'],
        algo=rand.suggest,
        rstate=np.random.RandomState(int(y[:].sum()) % 10000),
        early_stop_fn=stop,
        # The seed is deterministic but varies for each dataset and each split of it
        max_evals=1000)
    best = space_eval(param_grid_hyperopt['xgb'], best)

    clf = clf_(**best)
    clf.fit(x, y)

    if is_classification(metric_used):
        pred = clf.predict_proba(test_x)
    else:
        pred = clf.predict(test_x)
    metric = metric_used(test_y, pred)

    return metric, pred, best

"""
LEGACY UNUSED
"""

## Ridge
from sklearn.linear_model import RidgeClassifier
param_grid['ridge'] = {'alpha': [0, 0.1, .5, 1.0, 2.0], 'fit_intercept': [True, False]} # 'normalize': [False],
def ridge_metric(x, y, test_x, test_y, cat_features, metric_used):
    import warnings
    def warn(*args, **kwargs):
        pass

    warnings.warn = warn

    x, y, test_x, test_y = x.cpu(), y.cpu(), test_x.cpu(), test_y.cpu()
    x, test_x = torch.nan_to_num(x), torch.nan_to_num(test_x)

    clf = RidgeClassifier(n_jobs=MULTITHREAD)

    # create a dictionary of all values we want to test for n_neighbors
    # use gridsearch to test all values for n_neighbors
    clf = GridSearchCV(clf, param_grid['ridge'], cv=min(CV, x.shape[0]//2)
                       , scoring=get_scoring_string(metric_used)
                       , n_jobs=MULTITHREAD)
    # fit model to data
    clf.fit(x, y.long())

    pred = clf.decision_function(test_x)
    metric = metric_used(test_y, pred)

    return metric, pred

def mlp_acc(x, y, test_x, test_y, hyperparameters):
    num_layers, hidden_dim, activation_module, fixed_dropout_prob, is_binary_classification, epochs, lr, weight_decay = hyperparameters
    num_features = x.shape[1]

    x, y = x.to(device), y.to(device)
    test_x, test_y = test_x.to(device), test_y.to(device)

    def get_model():
        model = nn.Sequential(*[
            module for layer_idx in range(num_layers) for module in [
                nn.Linear(hidden_dim if layer_idx > 0 else num_features,
                          2 if layer_idx == num_layers - 1 else hidden_dim),
                torch.nn.Identity() if layer_idx == num_layers - 1 else activation_module(),
                torch.nn.Identity() if layer_idx == num_layers - 1 else torch.nn.Dropout(p=fixed_dropout_prob,
                                                                                         inplace=False)]
        ])
        if is_binary_classification:
            model.add_module(str(len(model)), torch.nn.Softmax(dim=1))  # TODO might also just do an round!?
        return model

    model = get_model().to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        # Forward pass
        y_pred = model(x)[:, 1]
        # Compute Loss

        loss = criterion(y_pred.squeeze(), y.float())

        # print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
        # Backward pass
        loss.backward()
        optimizer.step()

    model.eval()
    pred_y = model(test_x)[:, 1] > 0.5
    acc = (pred_y == test_y).float().mean()
    return acc

clf_dict = {'gp': gp_metric
, 'transformer': transformer_metric
, 'random_forest': random_forest_metric
                , 'knn': knn_metric
                , 'catboost': catboost_metric
                , 'tabnet': tabnet_metric
                , 'xgb': xgb_metric
                , 'lightgbm': lightgbm_metric
            , 'ridge': ridge_metric
                , 'logistic': logistic_metric
           , 'autosklearn': autosklearn_metric
             , 'autosklearn2': autosklearn2_metric
            , 'autogluon': autogluon_metric,
            'cocktail': well_tuned_simple_nets_metric}
