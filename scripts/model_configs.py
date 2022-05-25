from copy import deepcopy
from priors.utils import uniform_int_sampler_f
from priors.differentiable_prior import DifferentiableHyperparameter
from ConfigSpace import hyperparameters as CSH
import torch
from priors.differentiable_prior import replace_differentiable_distributions

import ConfigSpace as CS

def get_general_config(max_features, bptt, eval_positions=None):
    """"
    Returns the general PFN training hyperparameters.
    """
    config_general = {
        "lr": CSH.UniformFloatHyperparameter('lr', lower=0.00002, upper=0.0002, log=True),
        "dropout": CSH.CategoricalHyperparameter('dropout', [0.0]),
        "emsize": CSH.CategoricalHyperparameter('emsize', [2 ** i for i in range(8, 9)]), ## upper bound is -1
        "batch_size": CSH.CategoricalHyperparameter('batch_size', [2 ** i for i in range(8, 9)]),
        "nlayers": CSH.CategoricalHyperparameter('nlayers', [12]),
        "num_features": max_features,
        "nhead": CSH.CategoricalHyperparameter('nhead', [4]),
        "nhid_factor": 2,
        "bptt": bptt,
        "eval_positions": None,
        "seq_len_used": bptt,
        "sampling": 'normal',#hp.choice('sampling', ['mixed', 'normal']), # uniform
        "epochs": 80,
        "num_steps": 100,
        "verbose": False,
        "pre_sample_causes": True, # This is MLP
        "mix_activations": False,#hp.choice('mix_activations', [True, False]),
    }

    return config_general

def get_flexible_categorical_config(max_features):
    """"
    Returns the configuration parameters for the tabular multiclass wrapper.
    """
    config_flexible_categorical = {
        "nan_prob_unknown_reason_reason_prior": CSH.CategoricalHyperparameter('nan_prob_unknown_reason_reason_prior', [1.0]),
        "categorical_feature_p": CSH.CategoricalHyperparameter('categorical_feature_p', [0.0]),
        "nan_prob_no_reason": CSH.CategoricalHyperparameter('nan_prob_no_reason', [0.0, 0.1, 0.2]),
        "nan_prob_unknown_reason": CSH.CategoricalHyperparameter('nan_prob_unknown_reason', [0.0]),
        "nan_prob_a_reason": CSH.CategoricalHyperparameter('nan_prob_a_reason', [0.0]),
        # "num_classes": lambda : random.randint(2, 10), "balanced": False,
        "max_num_classes": 2,
        "num_classes": 2,
        "noise_type": CSH.CategoricalHyperparameter('noise_type', ["Gaussian"]), # NN
        "balanced": True,
        "normalize_to_ranking": CSH.CategoricalHyperparameter('normalize_to_ranking', [False]),
        "set_value_to_nan": CSH.CategoricalHyperparameter('set_value_to_nan', [0.5, 0.2, 0.0]),
        "normalize_by_used_features": True,
        "num_features_used":
            {'uniform_int_sampler_f(3,max_features)': uniform_int_sampler_f(1, max_features)}
        # hp.choice('conv_activation', [{'distribution': 'uniform', 'min': 2.0, 'max': 8.0}, None]),
    }
    return config_flexible_categorical

def get_diff_flex():
    """"
    Returns the configuration parameters for a differentiable wrapper around the tabular multiclass wrapper.
    """
    diff_flex = {
        # "ordinal_pct": {'distribution': 'uniform', 'min': 0.0, 'max': 0.5},
        # "num_categorical_features_sampler_a": hp.choice('num_categorical_features_sampler_a',
        #                                                 [{'distribution': 'uniform', 'min': 0.3, 'max': 0.9}, None]),
        # "num_categorical_features_sampler_b": {'distribution': 'uniform', 'min': 0.3, 'max': 0.9},
        "output_multiclass_ordered_p": {'distribution': 'uniform', 'min': 0.0, 'max': 0.5}, #CSH.CategoricalHyperparameter('output_multiclass_ordered_p', [0.0, 0.1, 0.2]),
        "multiclass_type": {'distribution': 'meta_choice', 'choice_values': ['value', 'rank']},
    }

    return diff_flex

def get_diff_gp():
    """"
    Returns the configuration parameters for a differentiable wrapper around GP.
    """
    diff_gp = {
        'outputscale': {'distribution': 'meta_trunc_norm_log_scaled', 'max_mean': 10., 'min_mean': 0.00001, 'round': False,
                        'lower_bound': 0},
        'lengthscale': {'distribution': 'meta_trunc_norm_log_scaled', 'max_mean': 10., 'min_mean': 0.00001, 'round': False,
                        'lower_bound': 0},
        'noise': {'distribution': 'meta_choice', 'choice_values': [0.00001, 0.0001, 0.01]}
    }

    return diff_gp

def get_diff_causal():
    """"
    Returns the configuration parameters for a differentiable wrapper around MLP / Causal mixture.
    """
    diff_causal = {
        "num_layers": {'distribution': 'meta_trunc_norm_log_scaled', 'max_mean': 6, 'min_mean': 1, 'round': True,
                       'lower_bound': 2},
        # Better beta?
        "prior_mlp_hidden_dim": {'distribution': 'meta_trunc_norm_log_scaled', 'max_mean': 130, 'min_mean': 5,
                                 'round': True, 'lower_bound': 4},

        "prior_mlp_dropout_prob": {'distribution': 'meta_beta', 'scale': 0.9, 'min': 0.1, 'max': 5.0},
    # This mustn't be too high since activations get too large otherwise

        "noise_std": {'distribution': 'meta_trunc_norm_log_scaled', 'max_mean': .3, 'min_mean': 0.0001, 'round': False,
                      'lower_bound': 0.0},
        "init_std": {'distribution': 'meta_trunc_norm_log_scaled', 'max_mean': 10.0, 'min_mean': 0.01, 'round': False,
                     'lower_bound': 0.0},
        "num_causes": {'distribution': 'meta_trunc_norm_log_scaled', 'max_mean': 12, 'min_mean': 1, 'round': True,
                       'lower_bound': 1},
        "is_causal": {'distribution': 'meta_choice', 'choice_values': [True, False]},
        "pre_sample_weights": {'distribution': 'meta_choice', 'choice_values': [True, False]},
        "y_is_effect": {'distribution': 'meta_choice', 'choice_values': [True, False]},
        "prior_mlp_activations": {'distribution': 'meta_choice_mixed', 'choice_values': [
            torch.nn.Tanh
            , torch.nn.ReLU
            , torch.nn.Identity
            , lambda : torch.nn.LeakyReLU(negative_slope=0.1)
            , torch.nn.ELU
        ]},
        "block_wise_dropout": {'distribution': 'meta_choice', 'choice_values': [True, False]},
        "sort_features": {'distribution': 'meta_choice', 'choice_values': [True, False]},
        "in_clique": {'distribution': 'meta_choice', 'choice_values': [True, False]},
    }

    return diff_causal

def get_diff_prior_bag():
    """"
    Returns the configuration parameters for a GP and MLP / Causal mixture.
    """
    diff_prior_bag = {
        'prior_bag_exp_weights_1': {'distribution': 'uniform', 'min': 100000., 'max': 100001.},
        # MLP Weight (Biased, since MLP works better, 1.0 is weight for prior number 0)
    }

    return diff_prior_bag

def get_diff_config():
    """"
    Returns the configuration parameters for a differentiable wrapper around GP and MLP / Causal mixture priors.
    """
    diff_prior_bag = get_diff_prior_bag()
    diff_causal = get_diff_causal()
    diff_gp = get_diff_gp()
    diff_flex = get_diff_flex()

    config_diff = {'differentiable_hyperparameters': {**diff_prior_bag, **diff_causal, **diff_gp, **diff_flex}}

    return config_diff


def sample_differentiable(config):
    """"
    Returns sampled hyperparameters from a differentiable wrapper, that is it makes a non-differentiable out of
    differentiable.
    """
    # config is a dict of dicts, dicts that have a 'distribution' key are treated as distributions to be sampled
    result = deepcopy(config)
    del result['differentiable_hyperparameters']

    for k, v in config['differentiable_hyperparameters'].items():
        s_indicator, s_hp = DifferentiableHyperparameter(**v, embedding_dim=None,
                                                         device=None)()  # both of these are actually not used to the best of my knowledge
        result[k] = s_hp

    return result

def list_all_hps_in_nested(config):
    """"
    Returns a list of hyperparameters from a neszed dict of hyperparameters.
    """

    if isinstance(config, CSH.Hyperparameter):
        return [config]
    elif isinstance(config, dict):
        result = []
        for k, v in config.items():
            result += list_all_hps_in_nested(v)
        return result
    else:
        return []

def create_configspace_from_hierarchical(config):
    cs = CS.ConfigurationSpace()
    for hp in list_all_hps_in_nested(config):
        cs.add_hyperparameter(hp)
    return cs

def fill_in_configsample(config, configsample):
    # config is our dict that defines config distribution
    # configsample is a CS.Configuration
    hierarchical_configsample = deepcopy(config)
    for k, v in config.items():
        if isinstance(v, CSH.Hyperparameter):
            hierarchical_configsample[k] = configsample[v.name]
        elif isinstance(v, dict):
            hierarchical_configsample[k] = fill_in_configsample(v, configsample)
    return hierarchical_configsample


def evaluate_hypers(config, sample_diff_hps=False):
    """"
    Samples a hyperparameter configuration from a sampleable configuration (can be used in HP search).
    """
    if sample_diff_hps:
        # I do a deepcopy here, such that the config stays the same and can still be used with diff. hps
        config = deepcopy(config)
        replace_differentiable_distributions(config)
    cs = create_configspace_from_hierarchical(config)
    cs_sample = cs.sample_configuration()
    return fill_in_configsample(config, cs_sample)
