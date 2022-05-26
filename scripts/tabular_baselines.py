from catboost import CatBoostClassifier, Pool

import math

from sklearn.impute import SimpleImputer

import xgboost as xgb
from sklearn import neighbors
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
import numpy as np

from scripts import tabular_metrics
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import time

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials , space_eval, rand
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

import autosklearn.classification

CV = 5
MULTITHREAD = 1 # Number of threads baselines are able to use at most
param_grid, param_grid_hyperopt = {}, {}

def get_scoring_direction(metric_used):
    # Not needed
    if metric_used == tabular_metrics.auc_metric:
        return -1
    elif metric_used == tabular_metrics.cross_entropy:
        return 1
    else:
        raise Exception('No scoring string found for metric')

def get_scoring_string(metric_used, multiclass=True, usage="sklearn_cv"):
    if metric_used == tabular_metrics.auc_metric:
        if usage == 'sklearn_cv':
            return 'roc_auc_ovo'
        elif usage == 'autogluon':
            return 'log_loss' # Autogluon crashes when using 'roc_auc' with some datasets usning logloss gives better scores;
                              # We might be able to fix this, but doesn't work out of box.
                              # File bug report? Error happens with dataset robert and fabert
            if multiclass:
                return 'roc_auc_ovo_macro'
            else:
                return 'roc_auc'
        elif usage == 'autosklearn':
            if multiclass:
                return autosklearn.metrics.log_loss # roc_auc only works for binary, use logloss instead
            else:
                return autosklearn.metrics.roc_auc
        elif usage == 'catboost':
            return 'MultiClass' # Effectively LogLoss, ROC not available
        elif usage == 'xgb':
            return 'logloss'
        return 'roc_auc'
    elif metric_used == tabular_metrics.cross_entropy:
        if usage == 'sklearn_cv':
            return 'neg_log_loss'
        elif usage == 'autogluon':
            return 'log_loss'
        elif usage == 'autosklearn':
            return autosklearn.metrics.log_loss
        elif usage == 'catboost':
            return 'MultiClass' # Effectively LogLoss
        return 'logloss'
    else:
        raise Exception('No scoring string found for metric')

def eval_f(params, clf_, x, y, metric_used, start_time, max_time):
    if time.time() - start_time > max_time:
        return np.nan
    scores = cross_val_score(clf_(**params), x, y, cv=CV, scoring=get_scoring_string(metric_used))

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

## Auto Gluon
def autogluon_metric(x, y, test_x, test_y, cat_features, metric_used, max_time=300):
    from autogluon.tabular import TabularPredictor # Inside function so package can be sued without installation
    x, y, test_x, test_y = preprocess_impute(x, y, test_x, test_y
                                             , one_hot=False
                                             , cat_features=cat_features
                                             , impute=False
                                             , standardize=False)
    train_data = pd.DataFrame(np.concatenate([x, y[:, np.newaxis]], 1))
    test_data = pd.DataFrame(np.concatenate([test_x, test_y[:, np.newaxis]], 1))

    # AutoGluon automatically infers datatypes, we don't specify the categorical labels
    predictor = TabularPredictor(
        label=train_data.columns[-1],
        eval_metric=get_scoring_string(metric_used, usage='autogluon', multiclass=(len(np.unique(y)) > 2)),
        problem_type='multiclass' if len(np.unique(y)) > 2 else 'binary'
        ## seed=int(y[:].sum()) doesn't accept seed
    ).fit(
        train_data=train_data,
        time_limit=max_time,
        presets=['best_quality']
        # The seed is deterministic but varies for each dataset and each split of it
    )

    pred = predictor.predict_proba(test_data, as_multiclass=True).values

    metric = metric_used(test_y, pred)

    return metric, pred, predictor.fit_summary()

## AUTO Sklearn
def autosklearn_metric(x, y, test_x, test_y, cat_features, metric_used, max_time=300):
    return autosklearn2_metric(x, y, test_x, test_y, cat_features, metric_used, max_time=max_time, version=1)

from autosklearn.experimental.askl2 import AutoSklearn2Classifier
from autosklearn.classification import AutoSklearnClassifier
def autosklearn2_metric(x, y, test_x, test_y, cat_features, metric_used, max_time=300, version=2):
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

    clf_ = AutoSklearn2Classifier if version == 2 else AutoSklearnClassifier
    clf = clf_(time_left_for_this_task=max_time,
                                                           memory_limit=4000,
                                                           n_jobs=MULTITHREAD,
               seed=int(y[:].sum()),
        # The seed is deterministic but varies for each dataset and each split of it
               metric=get_scoring_string(metric_used, usage='autosklearn', multiclass=len(np.unique(y)) > 2))

    # fit model to data
    clf.fit(x, y)

    pred = clf.predict_proba(test_x)
    metric = metric_used(test_y, pred)

    return metric, pred, None

param_grid_hyperopt['logistic'] = {
    'penalty': hp.choice('penalty', ['l1', 'l2', 'none'])
    , 'max_iter': hp.randint('max_iter', [50, 500])
    , 'fit_intercept': hp.choice('fit_intercept', [True, False])
    , 'C': hp.loguniform('C', -5, math.log(5.0))}  # 'normalize': [False],

def logistic_metric(x, y, test_x, test_y, cat_features, metric_used, max_time=300):
    x, y, test_x, test_y = preprocess_impute(x, y, test_x, test_y
                                             , one_hot=True, impute=True, standardize=True
                                             , cat_features=cat_features)

    def clf_(**params):
        return LogisticRegression(solver='saga', tol=1e-4, n_jobs=1, **params)

    start_time = time.time()

    def stop(trial):
        return time.time() - start_time > max_time, []

    best = fmin(
        fn=lambda params: eval_f(params, clf_, x, y, metric_used, start_time, max_time),
        space=param_grid_hyperopt['logistic'],
        algo=rand.suggest,
        rstate=np.random.RandomState(int(y[:].sum())),
        early_stop_fn=stop,
        # The seed is deterministic but varies for each dataset and each split of it
        max_evals=10000)
    best = space_eval(param_grid_hyperopt['logistic'], best)

    clf = clf_(**best)
    clf.fit(x, y)

    pred = clf.predict_proba(test_x)
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
        return neighbors.KNeighborsClassifier(n_jobs=1, **params)

    start_time = time.time()

    def stop(trial):
        return time.time() - start_time > max_time, []

    best = fmin(
        fn=lambda params: eval_f(params, clf_, x, y, metric_used, start_time, max_time),
        space=param_grid_hyperopt['knn'],
        algo=rand.suggest,
        rstate=np.random.RandomState(int(y[:].sum())),
        early_stop_fn=stop,
        # The seed is deterministic but varies for each dataset and each split of it
        max_evals=10000)
    best = space_eval(param_grid_hyperopt['knn'], best)

    clf = clf_(**best)
    clf.fit(x, y)

    pred = clf.predict_proba(test_x)
    metric = metric_used(test_y, pred)

    return metric, pred, best

## GP
param_grid_hyperopt['gp'] = {
    'params_y_scale': hp.loguniform('params_y_scale', math.log(0.05), math.log(5.0)),
    'params_length_scale': hp.loguniform('params_length_scale', math.log(0.1), math.log(1.0)),
    'n_jobs': hp.choice('njobs', [1])
}
def gp_metric(x, y, test_x, test_y, cat_features, metric_used, max_time=300):
    x, y, test_x, test_y = preprocess_impute(x, y, test_x, test_y,
                                             one_hot=True, impute=True, standardize=True,
                                             cat_features=cat_features)

    def clf_(params_y_scale,params_length_scale, **params):
        return GaussianProcessClassifier(kernel= params_y_scale * RBF(params_length_scale), **params)

    start_time = time.time()
    def stop(trial):
        return time.time() - start_time > max_time, []


    best = fmin(
        fn=lambda params: eval_f(params, clf_, x, y, metric_used, start_time, max_time),
        space=param_grid_hyperopt['gp'],
        algo=rand.suggest,
        rstate=np.random.RandomState(int(y[:].sum())),
        early_stop_fn=stop,
        # The seed is deterministic but varies for each dataset and each split of it
        max_evals=1000)
    best = space_eval(param_grid_hyperopt['gp'], best)

    clf = clf_(**best)
    clf.fit(x, y)

    pred = clf.predict_proba(test_x)
    metric = metric_used(test_y, pred)

    return metric, pred, best


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
    print(x)

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
        return CatBoostClassifier(
                               loss_function=get_scoring_string(metric_used, usage='catboost'),
                               thread_count = MULTITHREAD,
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
        rstate=np.random.RandomState(int(y[:].sum())),
        early_stop_fn=stop,
        # The seed is deterministic but varies for each dataset and each split of it
        max_evals=1000)
    best = space_eval(param_grid_hyperopt['catboost'], best)

    clf = clf_(**best)
    clf.fit(x, y)

    pred = clf.predict_proba(test_x)
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
    # XGB Documentation:
    # XGB handles categorical data appropriately without using One Hot Encoding, categorical features are experimetal
    # XGB handles missing values appropriately without imputation

    x, y, test_x, test_y = preprocess_impute(x, y, test_x, test_y
                                             , one_hot=False
                                             , cat_features=cat_features
                                             , impute=False
                                             , standardize=False)

    def clf_(**params):
        return xgb.XGBClassifier(use_label_encoder=False
                                 , nthread=1
                                 , **params
                                 , eval_metric=get_scoring_string(metric_used, usage='xgb') # AUC not implemented
        )

    start_time = time.time()
    def stop(trial):
        return time.time() - start_time > max_time, []

    best = fmin(
        fn=lambda params: eval_f(params, clf_, x, y, metric_used, start_time, max_time),
        space=param_grid_hyperopt['xgb'],
        algo=rand.suggest,
        rstate=np.random.RandomState(int(y[:].sum())),
        early_stop_fn=stop,
        # The seed is deterministic but varies for each dataset and each split of it
        max_evals=1000)
    best = space_eval(param_grid_hyperopt['xgb'], best)

    clf = clf_(**best)
    clf.fit(x, y)

    pred = clf.predict_proba(test_x)
    metric = metric_used(test_y, pred)

    return metric, pred, best


clf_dict = {'gp': gp_metric
                , 'knn': knn_metric
                , 'catboost': catboost_metric
                , 'xgb': xgb_metric
                , 'logistic': logistic_metric
           , 'autosklearn': autosklearn_metric
             , 'autosklearn2': autosklearn2_metric
            , 'autogluon': autogluon_metric}