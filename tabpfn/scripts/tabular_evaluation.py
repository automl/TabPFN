import time
import os
from pathlib import Path
from contextlib import nullcontext

import torch
from tqdm import tqdm
import random
import numpy as np

from torch import nn

from torch.utils.checkpoint import checkpoint
from tabpfn.utils import normalize_data, torch_nanmean, to_ranking_low_mem, remove_outliers
from tabpfn.scripts.tabular_baselines import get_scoring_string
from tabpfn.scripts import tabular_metrics
from tabpfn.scripts.transformer_prediction_interface import *
from tabpfn.scripts.baseline_prediction_interface import *
"""
===============================
PUBLIC FUNCTIONS FOR EVALUATION
===============================
"""


def eval_model(i, e, valid_datasets, test_datasets, eval_positions, bptt, add_name, base_path, device='cpu', eval_addition='', **kwargs):
    metrics_test, config_sample, model_path = eval_model_on_ds(i, e, test_datasets, eval_positions, bptt, add_name, base_path, device=device, eval_addition=eval_addition, **kwargs)
    metrics_valid, _, _ = eval_model_on_ds(i, e, valid_datasets, eval_positions, bptt, add_name, base_path, device=device, eval_addition=eval_addition, **kwargs)
    return {'mean_auc_test': metrics_test['mean_roc_at_1000'], 'mean_auc_valid': metrics_valid['mean_roc_at_1000'], 'mean_ce_test': metrics_test['mean_ce_at_1000'], 'mean_ce_valid': metrics_valid['mean_ce_at_1000'], 'config_sample': config_sample, 'model_path': model_path}

def eval_model_on_ds(i, e, valid_datasets, eval_positions, bptt, add_name, base_path, device='cpu', eval_addition='', **kwargs):

    # How to use: evaluate_without_fitting(i,0,valid_datasets, [1024], 100000, add_name=model_string, base_path=base_path,)
    def check_file(e):
        model_file = f'models_diff/prior_diff_real_checkpoint{add_name}_n_{i}_epoch_{e}.cpkt'
        model_path = os.path.join(base_path, model_file)
        # print('Evaluate ', model_path)
        results_file = os.path.join(base_path,
                                    f'models_diff/prior_diff_real_results{add_name}_n_{i}_epoch_{e}_{eval_addition}.pkl')
        if not Path(model_path).is_file():  # or Path(results_file).is_file():
            # print('checkpoint exists: ', Path(model_file).is_file(), ', results are written:', Path(results_file).is_file())
            return None, None, None
        return model_file, model_path, results_file

    if e == -1: # use last checkpoint, if e == -1
        for e_ in range(100, -1, -1):
            model_file_, model_path_, results_file_ = check_file(e_)
            if model_file_ is not None:
                e = e_
                model_file, model_path, results_file = model_file_, model_path_, results_file_
                break
    else:
        model_file, model_path, results_file = check_file(e)

    model, config_sample = load_model(base_path, model_file, device, None, verbose=False)

    params = {'max_features': config_sample['num_features']
        , 'rescale_features': config_sample["normalize_by_used_features"]
        , 'normalize_to_ranking': config_sample["normalize_to_ranking"]
        , 'normalize_with_sqrt': config_sample.get("normalize_with_sqrt", False)
              }
    metrics_valid = evaluate(datasets=valid_datasets, model=model[2], method='transformer', device=device, overwrite=True,
                             extend_features=True
                             # just removed the style keyword but transformer is trained with style, just empty
                             , save=False
                             , metric_used=tabular_metrics.cross_entropy
                             , return_tensor=True
                             , verbose=False
                             , eval_positions=eval_positions
                             , bptt=bptt
                             , base_path=None
                             , inference_mode=True
                             , **params
                             , **kwargs)

    tabular_metrics.calculate_score_per_method(tabular_metrics.auc_metric, 'roc', metrics_valid, valid_datasets, eval_positions)
    tabular_metrics.calculate_score_per_method(tabular_metrics.cross_entropy, 'ce', metrics_valid, valid_datasets, eval_positions)

    return metrics_valid, config_sample, model_path


def evaluate(datasets, bptt, eval_positions, metric_used, model, device='cpu'
             , verbose=False
             , return_tensor=False
             , **kwargs):
    """
    Evaluates a list of datasets for a model function.

    :param datasets: List of datasets
    :param bptt: maximum sequence length
    :param eval_positions: List of positions where to evaluate models
    :param verbose: If True, is verbose.
    :param metric_used: Which metric is optimized for.
    :param return_tensor: Wheater to return results as a pytorch.tensor or numpy, this is only relevant for transformer.
    :param kwargs:
    :return:
    """
    overall_result = {'metric_used': get_scoring_string(metric_used)
                      , 'bptt': bptt
                      , 'eval_positions': eval_positions}

    aggregated_metric_datasets, num_datasets = torch.tensor(0.0), 0

    # For each dataset
    for [ds_name, X, y, categorical_feats, _, _] in datasets:
        dataset_bptt = min(len(X), bptt)
        #if verbose and dataset_bptt < bptt:
        #    print(f'Dataset too small for given bptt, reducing to {len(X)} ({bptt})')

        aggregated_metric, num = torch.tensor(0.0), 0
        ds_result = {}

        for eval_position in (eval_positions if verbose else eval_positions):
            eval_position_real = int(dataset_bptt * 0.5) if 2 * eval_position > dataset_bptt else eval_position
            eval_position_bptt = int(eval_position_real * 2.0)

            r = evaluate_position(X, y, model=model
                        , num_classes=len(torch.unique(y))
                        , categorical_feats = categorical_feats
                        , bptt = eval_position_bptt
                        , ds_name=ds_name
                        , eval_position = eval_position_real
                        , metric_used = metric_used
                                  , device=device
                        ,**kwargs)

            if r is None:
                print('Execution failed')
                continue

            _, outputs, ys, best_configs, time_used = r

            if torch.is_tensor(outputs):
                outputs = outputs.to(outputs.device)
                ys = ys.to(outputs.device)

            # WARNING: This leaks information on the scaling of the labels
            if isinstance(model, nn.Module) and "BarDistribution" in str(type(model.criterion)):
                ys = (ys - torch.min(ys, axis=0)[0]) / (torch.max(ys, axis=0)[0] - torch.min(ys, axis=0)[0])

            # If we use the bar distribution and the metric_used is r2 -> convert buckets
            #  metric used is prob -> keep
            if isinstance(model, nn.Module) and "BarDistribution" in str(type(model.criterion)) and (
                    metric_used == tabular_metrics.r2_metric or metric_used == tabular_metrics.root_mean_squared_error_metric):
                ds_result[f'{ds_name}_bar_dist_at_{eval_position}'] = outputs
                outputs = model.criterion.mean(outputs)

            ys = ys.T
            ds_result[f'{ds_name}_best_configs_at_{eval_position}'] = best_configs
            ds_result[f'{ds_name}_outputs_at_{eval_position}'] = outputs
            ds_result[f'{ds_name}_ys_at_{eval_position}'] = ys
            ds_result[f'{ds_name}_time_at_{eval_position}'] = time_used

            new_metric = torch_nanmean(torch.stack([metric_used(ys[i], outputs[i]) for i in range(ys.shape[0])]))

            if not return_tensor:
                make_scalar = lambda x: float(x.detach().cpu().numpy()) if (torch.is_tensor(x) and (len(x.shape) == 0)) else x
                new_metric = make_scalar(new_metric)
                ds_result = {k: make_scalar(ds_result[k]) for k in ds_result.keys()}

            lib = torch if return_tensor else np
            if not lib.isnan(new_metric).any():
                aggregated_metric, num = aggregated_metric + new_metric, num + 1

        overall_result.update(ds_result)
        if num > 0:
            aggregated_metric_datasets, num_datasets = (aggregated_metric_datasets + (aggregated_metric / num)), num_datasets + 1

    overall_result['mean_metric'] = aggregated_metric_datasets / num_datasets

    return overall_result

"""
===============================
INTERNAL HELPER FUNCTIONS
===============================
"""

def check_file_exists(path):
    """Checks if a pickle file exists. Returns None if not, else returns the unpickled file."""
    if (os.path.isfile(path)):
        print(f'loading results from {path}')
        with open(path, 'rb') as f:
            return np.load(f, allow_pickle=True).tolist()
    return None

def generate_valid_split(X, y, bptt, eval_position, is_classification, split_number=1):
    """Generates a deteministic train-(test/valid) split. Both splits must contain the same classes and all classes in
    the entire datasets. If no such split can be sampled in 7 passes, returns None.

    :param X: torch tensor, feature values
    :param y: torch tensor, class values
    :param bptt: Number of samples in train + test
    :param eval_position: Number of samples in train, i.e. from which index values are in test
    :param split_number: The split id
    :return:
    """
    done, seed = False, 13

    torch.manual_seed(split_number)
    perm = torch.randperm(X.shape[0]) if split_number > 1 else torch.arange(0, X.shape[0])
    X, y = X[perm], y[perm]
    while not done:
        if seed > 20:
            return None, None # No split could be generated in 7 passes, return None
        random.seed(seed)
        i = random.randint(0, len(X) - bptt) if len(X) - bptt > 0 else 0
        y_ = y[i:i + bptt]

        if is_classification:
            # Checks if all classes from dataset are contained and classes in train and test are equal (contain same
            # classes) and
            done = len(torch.unique(y_)) == len(torch.unique(y))
            done = done and torch.all(torch.unique(y_) == torch.unique(y))
            done = done and len(torch.unique(y_[:eval_position])) == len(torch.unique(y_[eval_position:]))
            done = done and torch.all(torch.unique(y_[:eval_position]) == torch.unique(y_[eval_position:]))
            seed = seed + 1
        else:
            done = True

    eval_xs = torch.stack([X[i:i + bptt].clone()], 1)
    eval_ys = torch.stack([y[i:i + bptt].clone()], 1)

    return eval_xs, eval_ys


def evaluate_position(X, y, categorical_feats, model, bptt
                      , eval_position, overwrite, save, base_path, path_interfix, method, ds_name, fetch_only=False
                      , max_time=300, split_number=1, metric_used=None, device='cpu'
                      , per_step_normalization=False, **kwargs):
    """
    Evaluates a dataset with a 'bptt' number of training samples.

    :param X: Dataset X
    :param y: Dataset labels
    :param categorical_feats: Indices of categorical features.
    :param model: Model function
    :param bptt: Sequence length.
    :param eval_position: Number of training samples.
    :param overwrite: Wheater to ove
    :param overwrite: If True, results on disk are overwritten.
    :param save:
    :param path_interfix: Used for constructing path to write on disk.
    :param method: Model name.
    :param ds_name: Datset name.
    :param fetch_only: Wheater to calculate or only fetch results.
    :param per_step_normalization:
    :param kwargs:
    :return:
    """

    if save:
        path = os.path.join(base_path, f'results/tabular/{path_interfix}/results_{method}_{ds_name}_{eval_position}_{bptt}_{split_number}.npy')
        #log_path =

    ## Load results if on disk
    if not overwrite:
        result = check_file_exists(path)
        if result is not None:
            if not fetch_only:
                print(f'Loaded saved result for {path}')
            return result
        elif fetch_only:
            print(f'Could not load saved result for {path}')
            return None

    ## Generate data splits
    eval_xs, eval_ys = generate_valid_split(X, y, bptt, eval_position
                                            , is_classification=tabular_metrics.is_classification(metric_used)
                                            , split_number=split_number)
    if eval_xs is None:
        print(f"No dataset could be generated {ds_name} {bptt}")
        return None

    eval_ys = (eval_ys > torch.unique(eval_ys).unsqueeze(0)).sum(axis=1).unsqueeze(-1)

    if isinstance(model, nn.Module):
        model = model.to(device)
        eval_xs = eval_xs.to(device)
        eval_ys = eval_ys.to(device)

    start_time = time.time()

    if isinstance(model, nn.Module): # Two separate predict interfaces for transformer and baselines
        outputs, best_configs = transformer_predict(model, eval_xs, eval_ys, eval_position, metric_used=metric_used
                                                    , categorical_feats=categorical_feats
                                                    , inference_mode=True
                                                    , device=device
                                                    , extend_features=True,
                                                    **kwargs), None
    else:
        _, outputs, best_configs = baseline_predict(model, eval_xs, eval_ys, categorical_feats
                                                    , eval_pos=eval_position
                                                    , device=device
                                                    , max_time=max_time, metric_used=metric_used, **kwargs)
    eval_ys = eval_ys[eval_position:]
    if outputs is None:
        print('Execution failed')
        return None

    if torch.is_tensor(outputs): # Transfers data to cpu for saving
        outputs = outputs.cpu()
        eval_ys = eval_ys.cpu()

    ds_result = None, outputs, eval_ys, best_configs, time.time() - start_time

    if save:
        with open(path, 'wb') as f:
            np.save(f, ds_result)
            print(f'saved results to {path}')

    return ds_result