import os
import torch
import numpy as np
import time
import pickle
from tabpfn.scripts import  tabular_metrics
from tabpfn.scripts.tabular_metrics import calculate_score_per_method
from tabpfn.scripts.tabular_evaluation import evaluate
from tqdm import tqdm
import random
from tabpfn.scripts.transformer_prediction_interface import get_params_from_config
from tabpfn.scripts.transformer_prediction_interface import load_model_workflow

"""
===============================
PUBLIC FUNCTIONS FOR EVALUATION
===============================
"""


def eval_model_range(i_range, *args, **kwargs):
    for i in i_range:
        eval_model(i, *args, **kwargs)

def eval_model(i, e, valid_datasets, test_datasets, train_datasets, add_name, base_path,  eval_positions_valid=[1000], eval_positions_test=[1000],
               bptt_valid=2000,
               bptt_test=2000, device='cpu', eval_addition='', differentiable=False, **extra_tuning_args):
    """
    Differentiable model evaliation workflow. Evaluates and saves results to disk.

    :param i:
    :param e:
    :param valid_datasets:
    :param test_datasets:
    :param train_datasets:
    :param eval_positions_valid:
    :param eval_positions_test:
    :param bptt_valid:
    :param bptt_test:
    :param add_name:
    :param base_path:
    :param device:
    :param eval_addition:
    :param extra_tuning_args:
    :return:
    """
    model, c, results_file = load_model_workflow(i, e, add_name, base_path, device, eval_addition)
    params = {'bptt': bptt_valid
        , 'bptt_final': bptt_test
        , 'eval_positions': eval_positions_valid
        , 'eval_positions_test': eval_positions_test
        , 'valid_datasets': valid_datasets
        , 'test_datasets': test_datasets
        , 'train_datasets': train_datasets
        , 'verbose': True
        , 'device': device
              }

    params.update(get_params_from_config(c))

    start = time.time()
    metrics, metrics_valid, style, temperature, optimization_route = evaluate_point_model(model, **params,
                                                                                                       **extra_tuning_args)
    print('Evaluation time: ', time.time() - start)

    print(results_file)
    r = [c.copy(), metrics, metrics_valid, style.to('cpu') if style else style, temperature.to('cpu') if temperature else temperature, optimization_route]
    with open(results_file, 'wb') as output:
        del r[0]['num_features_used']
        del r[0]['categorical_features_sampler']
        pickle.dump(r, output)

    _, _, _, style, temperature, _ = r

    return r, model

"""
===============================
INTERNAL HELPER FUNCTIONS
===============================
"""

def evaluate_point_model(model
                                  , valid_datasets
                                  , test_datasets
                                  , train_datasets
                                  , eval_positions_test=None
                                  , bptt_final=200
                                  , device='cpu'
                                  , selection_metric='auc'
                                  , final_splits=[1, 2, 3, 4, 5]
                                  , N_ensemble_configurations_list=[1, 5, 10, 20, 50, 100]
                         , bptt=None
                         , eval_positions=None
                                  , **kwargs):
    """
    Evaluation function for diffable model evaluation. Returns a list of results.

    :param model:
    :param valid_datasets:
    :param test_datasets:
    :param train_datasets:
    :param N_draws:
    :param N_grad_steps:
    :param eval_positions:
    :param eval_positions_test:
    :param bptt:
    :param bptt_final:
    :param style:
    :param n_parallel_configurations:
    :param device:
    :param selection_metric:
    :param final_splits:
    :param N_ensemble_configurations_list:
    :param kwargs:
    :return:
    """
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    evaluation_metric = tabular_metrics.auc_metric
    selection_metric = tabular_metrics.auc_metric

    model[2].to(device)
    model[2].eval()

    def final_evaluation():
        print('Running eval dataset with final params (no gradients)..')
        result_test = []
        for N_ensemble_configurations in N_ensemble_configurations_list:
            print(f'Running with {N_ensemble_configurations} ensemble_configurations')
            kwargs['N_ensemble_configurations'] = N_ensemble_configurations
            splits = []
            for split in final_splits:
                splits += [eval_step(test_datasets, None, softmax_temperature=torch.tensor([0])
                                     , return_tensor=False, eval_positions=eval_positions_test,
                                     bptt=bptt_final, split_number=split, model=model[2], device=device
                                     , selection_metric=selection_metric, evaluation_metric=evaluation_metric
                                     , **kwargs)]
            result_test += [splits]

        print('Running valid dataset with final params (no gradients)..')
        result_valid = eval_step(valid_datasets, None, softmax_temperature=torch.tensor([0])
                                 , return_tensor=False, eval_positions=eval_positions_test,
                                 bptt=bptt_final, model=model[2], device=device
                                 , selection_metric=selection_metric, evaluation_metric=evaluation_metric,**kwargs)

        return result_test, result_valid

    result_test, result_valid = final_evaluation()

    return result_test, result_valid, None, None, None

def eval_step(ds, used_style, selection_metric, evaluation_metric, eval_positions, return_tensor=True, **kwargs):
    def step():
        return evaluate(datasets=ds,
                        method='transformer'
                        , overwrite=True
                        , style=used_style
                        , eval_positions=eval_positions
                        , metric_used=selection_metric
                        , save=False
                        , path_interfix=None
                        , base_path=None
                        , **kwargs)

    if return_tensor:
        r = step()
    else:
        with torch.no_grad():
            r = step()

    calculate_score_per_method(selection_metric, 'select', r, ds, eval_positions, aggregator='mean')
    calculate_score_per_method(evaluation_metric, 'eval', r, ds, eval_positions, aggregator='mean')

    return r


def gradient_optimize_style(model, init_style, steps, softmax_temperature, train_datasets, valid_datasets, bptt, learning_rate=0.03, optimize_all=False,
                            limit_style=True, N_datasets_sampled=90, optimize_softmax_temperature=True, selection_metric_min_max='max', **kwargs):
    """
    Uses gradient based methods to optimize 'style' on the 'train_datasets' and uses stopping with 'valid_datasets'.

    :param model:
    :param init_style:
    :param steps:
    :param learning_rate:
    :param softmax_temperature:
    :param train_datasets:
    :param valid_datasets:
    :param optimize_all:
    :param limit_style:
    :param N_datasets_sampled:
    :param optimize_softmax_temperature:
    :param selection_metric_min_max:
    :param kwargs:
    :return:
    """
    grad_style = torch.nn.Parameter(init_style.detach(), requires_grad=True)

    best_style, best_temperature, best_selection_metric, best_diffable_metric = grad_style.detach(), softmax_temperature.detach(), None, None
    softmax_temperature = torch.nn.Parameter(softmax_temperature.detach(), requires_grad=optimize_softmax_temperature)
    variables_to_optimize = model[2].parameters() if optimize_all else [grad_style, softmax_temperature]
    optimizer = torch.optim.Adam(variables_to_optimize, lr=learning_rate)

    optimization_route_selection, optimization_route_diffable = [], []
    optimization_route_selection_valid, optimization_route_diffable_valid = [], []

    def eval_opt(ds, return_tensor=True, inference_mode=False):
        result = eval_step(ds, grad_style, softmax_temperature=softmax_temperature, return_tensor=return_tensor
                           , inference_mode=inference_mode, model=model[2], bptt=bptt, **kwargs)

        diffable_metric = result['mean_metric']
        selection_metric = result['mean_select']

        return diffable_metric, selection_metric

    def eval_all_datasets(datasets, propagate=True):
        selection_metrics_this_step, diffable_metrics_this_step = [], []
        for ds in datasets:
            diffable_metric_train, selection_metric_train = eval_opt([ds], inference_mode=(not propagate))
            if not torch.isnan(diffable_metric_train).any():
                if propagate and diffable_metric_train.requires_grad == True:
                    diffable_metric_train.backward()
                selection_metrics_this_step += [selection_metric_train]
                diffable_metrics_this_step += [float(diffable_metric_train.detach().cpu().numpy())]
        diffable_metric_train = np.nanmean(diffable_metrics_this_step)
        selection_metric_train = np.nanmean(selection_metrics_this_step)

        return diffable_metric_train, selection_metric_train

    for t in tqdm(range(steps), desc='Iterate over Optimization steps'):
        optimizer.zero_grad()

        # Select subset of datasets
        random.seed(t)
        train_datasets_ = random.sample(train_datasets, N_datasets_sampled)

        # Get score on train
        diffable_metric_train, selection_metric_train = eval_all_datasets(train_datasets_, propagate=True)
        optimization_route_selection += [float(selection_metric_train)]
        optimization_route_diffable += [float(diffable_metric_train)]

        # Get score on valid
        diffable_metric_valid, selection_metric_valid = eval_all_datasets(valid_datasets, propagate=False)
        optimization_route_selection_valid += [float(selection_metric_valid)]
        optimization_route_diffable_valid += [float(diffable_metric_valid)]

        is_best = (best_selection_metric is None)
        is_best = is_best or (selection_metric_min_max == 'min' and best_selection_metric > selection_metric_valid)
        is_best = is_best or (selection_metric_min_max == 'max' and best_selection_metric < selection_metric_valid)
        if (not np.isnan(selection_metric_valid) and is_best):
            print('New best', best_selection_metric, selection_metric_valid)
            best_style = grad_style.detach().clone()
            best_temperature = softmax_temperature.detach().clone()
            best_selection_metric, best_diffable_metric = selection_metric_valid, diffable_metric_valid

        optimizer.step()

        if limit_style:
            grad_style = grad_style.detach().clamp(-1.74, 1.74)

        print(f'Valid: Diffable metric={diffable_metric_valid} Selection metric={selection_metric_valid};' +
            f'Train: Diffable metric={diffable_metric_train} Selection metric={selection_metric_train}')

    print(f'Return best:{best_style} {best_selection_metric}')
    return {'best_style': best_style, 'best_temperature': best_temperature
            , 'optimization_route': {'select': optimization_route_selection, 'loss': optimization_route_diffable,
               'test_select': optimization_route_selection_valid, 'test_loss': optimization_route_diffable_valid}}