import os
import pathlib
from argparse import Namespace

from sklearn.model_selection import GridSearchCV
import sys

CV = 5
param_grid = {}

param_grid['saint'] = {
    # as in https://github.com/kathrinse/TabSurvey/blob/main/models/saint.py#L268
    "dim": [32, 64, 128, 256],
    "depth": [1, 2, 3, 6, 12],
    "heads": [2, 4, 8],
    "dropout": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
}

def saint_metric(x, y, test_x, test_y, cat_features, metric_used):
    ## Original Implementation https://github.com/somepago/saint
    ## Reimplementation from https://github.com/kathrinse/TabSurvey
    ## HowTo install
    # git clone git@github.com:kathrinse/TabSurvey.git
    # cd TabSurvey
    # requirements
    # optuna
    # scikit-learn
    # pandas
    # configargparse
    # torch
    # einops
    pre_cwd = os.getcwd()

    # TODO: Make sure that we change to TabSurvey in here
    # Assume it is in ../../TabSurvey
    dest_wd = pathlib.Path(__file__).absolute().parent.parent.joinpath("../TabSurvey")
    print(f"Change from {pre_cwd} to {dest_wd}")
    sys.chdir(dest_wd)

    try:
        from models.saint import SAINT

        import warnings
        def warn(*args, **kwargs):
            pass

        # get cat dims
        # assume cat_features is a list of idx
        # TODO: FIX this if wrong
        cat_dims = []
        for idx in cat_features:
            cat_dims.append(len(set(x[idx, :])))
        model_args = Namespace(
            num_features=x.shape[1],
            cat_idx=cat_features,
            cat_dims=cat_dims,
        )
        warnings.warn = warn

        x, y, test_x, test_y = x.cpu(), y.cpu(), test_x.cpu(), test_y.cpu()

        clf = SAINT(model_args)

        clf = GridSearchCV(clf, param_grid['saint'], cv=min(CV, x.shape[0]//2))
        # fit model to data
        clf.fit(x, y.long())

        pred = clf.decision_function(test_x)
        metric = metric_used(test_y.cpu().numpy(), pred)
    except:
        raise
    finally:
        os.chdir(pre_cwd)
    return metric, pred