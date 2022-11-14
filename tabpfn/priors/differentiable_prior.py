import torch
from torch import nn
import math

from .utils import get_batch_to_dataloader
from tabpfn.utils import default_device, normalize_by_used_features_f
from .utils import order_by_y

from .utils import trunc_norm_sampler_f, beta_sampler_f, gamma_sampler_f, uniform_sampler_f, zipf_sampler_f, scaled_beta_sampler_f, uniform_int_sampler_f


def unpack_dict_of_tuples(d):
    # Returns list of dicts where each dict i contains values of tuple position i
    # {'a': (1,2), 'b': (3,4)} -> [{'a': 1, 'b': 3}, {'a': 2, 'b': 4}]
    return [dict(zip(d.keys(), v)) for v in list(zip(*list(d.values())))]

class DifferentiableHyperparameter(nn.Module):
    ## We can sample this and get a hyperparameter value and a normalized hyperparameter indicator
    def __init__(self, distribution, embedding_dim, device, **args):
        super(DifferentiableHyperparameter, self).__init__()

        self.distribution = distribution
        self.embedding_dim = embedding_dim
        self.device=device
        for key in args:
            setattr(self, key, args[key])

        def get_sampler():
            #if self.distribution == "beta":
            #    return beta_sampler_f(self.a, self.b), 0, 1
            #elif self.distribution == "gamma":
            #    return gamma_sampler_f(self.a, self.b), 0, 1
            #elif self.distribution == "beta_int":
            #    return scaled_beta_sampler_f(self.a, self.b, self.scale, self.min), self.scale + self.min, self.min, self.a / (self.a + self.b)
            if self.distribution == "uniform":
                if not hasattr(self, 'sample'):
                    return uniform_sampler_f(self.min, self.max), self.min, self.max, (self.max+self.min) / 2, math.sqrt(1/12*(self.max-self.min)*(self.max-self.min))
                else:
                    return lambda: self.sample, self.min, self.max, None, None
            elif self.distribution == "uniform_int":
                return uniform_int_sampler_f(self.min, self.max), self.min, self.max, (self.max+self.min) / 2, math.sqrt(1/12*(self.max-self.min)*(self.max-self.min))

        if self.distribution.startswith("meta"):
            self.hparams = {}
            def sample_meta(f):
                indicators, passed = unpack_dict_of_tuples({hp: self.hparams[hp]() for hp in self.hparams})
                # sampled_embeddings = list(itertools.chain.from_iterable([sampled_embeddings[k] for k in sampled_embeddings]))
                meta_passed = f(**passed)
                return indicators, meta_passed

            args_passed = {'device': device, 'embedding_dim': embedding_dim}
            if self.distribution == "meta_beta":
                ## Truncated normal where std and mean are drawn randomly logarithmically scaled
                if hasattr(self, 'b') and hasattr(self, 'k'):
                    self.hparams = {'b': lambda: (None, self.b), 'k': lambda: (None, self.k)}
                else:
                    self.hparams = {"b": DifferentiableHyperparameter(distribution="uniform", min=self.min
                                                                                          , max=self.max, **args_passed)
                                    , "k": DifferentiableHyperparameter(distribution="uniform", min=self.min
                                                                                           , max=self.max, **args_passed)}
                def make_beta(b, k):
                    return lambda b=b, k=k: self.scale * beta_sampler_f(b, k)()
                self.sampler = lambda make_beta=make_beta : sample_meta(make_beta)
            if self.distribution == "meta_gamma":
                ## Truncated normal where std and mean are drawn randomly logarithmically scaled
                if hasattr(self, 'alpha') and hasattr(self, 'scale'):
                    self.hparams = {'alpha': lambda: (None, self.alpha), 'scale': lambda: (None, self.scale)}
                else:
                    self.hparams = {"alpha": DifferentiableHyperparameter(distribution="uniform", min=0.0
                                                                                          , max=math.log(self.max_alpha), **args_passed)
                                    , "scale": DifferentiableHyperparameter(distribution="uniform", min=0.0
                                                                                           , max=self.max_scale, **args_passed)}
                def make_gamma(alpha, scale):
                    return lambda alpha=alpha, scale=scale: self.lower_bound + round(gamma_sampler_f(math.exp(alpha), scale / math.exp(alpha))()) if self.round else self.lower_bound + gamma_sampler_f(math.exp(alpha), scale / math.exp(alpha))()
                self.sampler = lambda make_gamma=make_gamma : sample_meta(make_gamma)
            elif self.distribution == "meta_trunc_norm_log_scaled":
                # these choices are copied down below, don't change these without changing `replace_differentiable_distributions`
                self.min_std = self.min_std if hasattr(self, 'min_std') else 0.01
                self.max_std = self.max_std if hasattr(self, 'max_std') else 1.0
                ## Truncated normal where std and mean are drawn randomly logarithmically scaled
                if not hasattr(self, 'log_mean'):
                    self.hparams = {"log_mean": DifferentiableHyperparameter(distribution="uniform", min=math.log(self.min_mean)
                                                                                          , max=math.log(self.max_mean), **args_passed)
                                    , "log_std": DifferentiableHyperparameter(distribution="uniform", min=math.log(self.min_std)
                                                                                           , max=math.log(self.max_std), **args_passed)}
                else:
                    self.hparams = {'log_mean': lambda: (None, self.log_mean), 'log_std': lambda: (None, self.log_std)}
                def make_trunc_norm(log_mean, log_std):
                    return ((lambda : self.lower_bound + round(trunc_norm_sampler_f(math.exp(log_mean), math.exp(log_mean)*math.exp(log_std))())) if self.round
                            else (lambda: self.lower_bound + trunc_norm_sampler_f(math.exp(log_mean), math.exp(log_mean)*math.exp(log_std))()))

                self.sampler = lambda make_trunc_norm=make_trunc_norm: sample_meta(make_trunc_norm)
            elif self.distribution == "meta_trunc_norm":
                self.min_std = self.min_std if hasattr(self, 'min_std') else 0.01
                self.max_std = self.max_std if hasattr(self, 'max_std') else 1.0
                self.hparams = {"mean": DifferentiableHyperparameter(distribution="uniform", min=self.min_mean
                                                                                      , max=self.max_mean, **args_passed)
                                , "std": DifferentiableHyperparameter(distribution="uniform", min=self.min_std
                                                                                       , max=self.max_std, **args_passed)}
                def make_trunc_norm(mean, std):
                    return ((lambda: self.lower_bound + round(
                        trunc_norm_sampler_f(mean, std)())) if self.round
                            else (
                        lambda make_trunc_norm=make_trunc_norm: self.lower_bound + trunc_norm_sampler_f(mean, std)()))
                self.sampler = lambda : sample_meta(make_trunc_norm)
            elif self.distribution == "meta_choice":
                if hasattr(self, 'choice_1_weight'):
                    self.hparams = {f'choice_{i}_weight': lambda: (None, getattr(self, f'choice_{i}_weight')) for i in range(1, len(self.choice_values))}
                else:
                    self.hparams = {f"choice_{i}_weight": DifferentiableHyperparameter(distribution="uniform", min=-3.0
                                                                                          , max=5.0, **args_passed) for i in range(1, len(self.choice_values))}
                def make_choice(**choices):
                    choices = torch.tensor([1.0] + [choices[i] for i in choices], dtype=torch.float)
                    weights = torch.softmax(choices, 0)  # create a tensor of weights
                    sample = torch.multinomial(weights, 1, replacement=True).numpy()[0]
                    return self.choice_values[sample]

                self.sampler = lambda make_choice=make_choice: sample_meta(make_choice)
            elif self.distribution == "meta_choice_mixed":
                if hasattr(self, 'choice_1_weight'):
                    self.hparams = {f'choice_{i}_weight': lambda: (None, getattr(self, f'choice_{i}_weight')) for i in range(1, len(self.choice_values))}
                else:
                    self.hparams = {f"choice_{i}_weight": DifferentiableHyperparameter(distribution="uniform", min=-5.0
                                                                                          , max=6.0, **args_passed) for i in range(1, len(self.choice_values))}
                def make_choice(**choices):
                    weights = torch.softmax(torch.tensor([1.0] + [choices[i] for i in choices], dtype=torch.float), 0)  # create a tensor of weights
                    def sample():
                        s = torch.multinomial(weights, 1, replacement=True).numpy()[0]
                        return self.choice_values[s]()
                    return lambda: sample

                self.sampler = lambda make_choice=make_choice: sample_meta(make_choice)
        else:
            def return_two(x, min, max, mean, std):
                # Returns (a hyperparameter value, and an indicator value passed to the model)
                if mean is not None:
                    ind = (x-mean)/std#(2 * (x-min) / (max-min) - 1)
                else:
                    ind = None
                return ind, x # normalize indicator to [-1, 1]
            # def sample_standard(sampler_f, embedding):
            #     s = torch.tensor([sampler_f()], device = self.device)
            #     return s, embedding(s)
            self.sampler_f, self.sampler_min, self.sampler_max, self.sampler_mean, self.sampler_std = get_sampler()
            self.sampler = lambda : return_two(self.sampler_f(), min=self.sampler_min, max=self.sampler_max
                                               , mean=self.sampler_mean, std=self.sampler_std)
            # self.embedding_layer = nn.Linear(1, self.embedding_dim, device=self.device)
            # self.embed = lambda x : self.embedding_layer(
            #     (x - self.sampler_min) / (self.sampler_max - self.sampler_min))
            #self.sampler = lambda : sample_standard(self.sampler_f, self.embedding)


    def forward(self):
        s, s_passed = self.sampler()
        return s, s_passed



class DifferentiableHyperparameterList(nn.Module):
    def __init__(self, hyperparameters, embedding_dim, device):
        super().__init__()

        self.device = device
        hyperparameters = {k: v for (k, v) in hyperparameters.items() if v}
        self.hyperparameters = nn.ModuleDict({hp: DifferentiableHyperparameter(embedding_dim = embedding_dim
                                                                               , name = hp
                                                                               , device = device, **hyperparameters[hp]) for hp in hyperparameters})
    def get_hyperparameter_info(self):
        sampled_hyperparameters_f, sampled_hyperparameters_keys = [], []
        def append_hp(hp_key, hp_val):
            sampled_hyperparameters_keys.append(hp_key)
            # Function remaps hyperparameters from [-1, 1] range to true value
            s_min, s_max, s_mean, s_std = hp_val.sampler_min, hp_val.sampler_max, hp_val.sampler_mean, hp_val.sampler_std
            sampled_hyperparameters_f.append((lambda x: (x-s_mean)/s_std, lambda y : (y * s_std)+s_mean))
            #sampled_hyperparameters_f.append(((lambda x: ((x - s_min) / (s_max - s_min) * (2) - 1)
            #                                  , (lambda y: ((y + 1) * (1 / 2) * (s_max - s_min) + s_min))))
        for hp in self.hyperparameters:
            hp_val = self.hyperparameters[hp]
            if hasattr(hp_val, 'hparams'):
                for hp_ in hp_val.hparams:
                    append_hp(f'{hp}_{hp_}', hp_val.hparams[hp_])
            else:
                append_hp(hp, hp_val)


        return sampled_hyperparameters_keys, sampled_hyperparameters_f

    def sample_parameter_object(self):
        sampled_hyperparameters, s_passed = {}, {}
        for hp in self.hyperparameters:
            sampled_hyperparameters_, s_passed_ = self.hyperparameters[hp]()
            s_passed[hp] = s_passed_
            if isinstance(sampled_hyperparameters_, dict):
                sampled_hyperparameters_ = {hp + '_' + str(key): val for key, val in sampled_hyperparameters_.items()}
                sampled_hyperparameters.update(sampled_hyperparameters_)
            else:
                sampled_hyperparameters[hp] = sampled_hyperparameters_

        # s_passed contains the values passed to the get_batch function
        # sampled_hyperparameters contains the indicator of the sampled value, i.e. only number that describe the sampled object
        return s_passed, sampled_hyperparameters#self.pack_parameter_object(sampled_embeddings)

class DifferentiablePrior(torch.nn.Module):
    def __init__(self, get_batch, hyperparameters, differentiable_hyperparameters, args):
        super(DifferentiablePrior, self).__init__()

        self.h = hyperparameters
        self.args = args
        self.get_batch = get_batch
        self.differentiable_hyperparameters = DifferentiableHyperparameterList(differentiable_hyperparameters
                                                                               , embedding_dim=self.h['emsize']
                                                                               , device=self.args['device'])

    def forward(self):
        # Sample hyperparameters
        sampled_hyperparameters_passed, sampled_hyperparameters_indicators = self.differentiable_hyperparameters.sample_parameter_object()

        hyperparameters = {**self.h, **sampled_hyperparameters_passed}
        x, y, y_ = self.get_batch(hyperparameters=hyperparameters, **self.args)

        return x, y, y_, sampled_hyperparameters_indicators


# TODO: Make this a class that keeps objects
@torch.no_grad()
def get_batch(batch_size, seq_len, num_features, get_batch
              , device=default_device, differentiable_hyperparameters={}
              , hyperparameters=None, batch_size_per_gp_sample=None, **kwargs):
    batch_size_per_gp_sample = batch_size_per_gp_sample or (min(64, batch_size))
    num_models = batch_size // batch_size_per_gp_sample
    assert num_models * batch_size_per_gp_sample == batch_size, f'Batch size ({batch_size}) not divisible by batch_size_per_gp_sample ({batch_size_per_gp_sample})'

    args = {'device': device, 'seq_len': seq_len, 'num_features': num_features, 'batch_size': batch_size_per_gp_sample}
    args = {**kwargs, **args}

    models = [DifferentiablePrior(get_batch, hyperparameters, differentiable_hyperparameters, args) for _ in range(num_models)]
    sample = sum([[model()] for model in models], [])

    x, y, y_, hyperparameter_dict = zip(*sample)

    if 'verbose' in hyperparameters and hyperparameters['verbose']:
        print('Hparams', hyperparameter_dict[0].keys())

    hyperparameter_matrix = []
    for batch in hyperparameter_dict:
        hyperparameter_matrix.append([batch[hp] for hp in batch])

    transposed_hyperparameter_matrix = list(zip(*hyperparameter_matrix))
    assert all([all([hp is None for hp in hp_]) or all([hp is not None for hp in hp_]) for hp_ in transposed_hyperparameter_matrix]), 'it should always be the case that when a hyper-parameter is None, once it is always None'
    # we remove columns that are only None (i.e. not sampled)
    hyperparameter_matrix = [[hp for hp in hp_ if hp is not None] for hp_ in hyperparameter_matrix]
    if len(hyperparameter_matrix[0]) > 0:
        packed_hyperparameters = torch.tensor(hyperparameter_matrix)
        packed_hyperparameters = torch.repeat_interleave(packed_hyperparameters, repeats=batch_size_per_gp_sample, dim=0).detach()
    else:
        packed_hyperparameters = None

    x, y, y_, packed_hyperparameters = (torch.cat(x, 1).detach()
                                        , torch.cat(y, 1).detach()
                                        , torch.cat(y_, 1).detach()
                                        , packed_hyperparameters)#list(itertools.chain.from_iterable(itertools.repeat(x, batch_size_per_gp_sample) for x in packed_hyperparameters)))#torch.repeat_interleave(torch.stack(packed_hyperparameters, 0).detach(), repeats=batch_size_per_gp_sample, dim=0))
    return x, y, y_, (packed_hyperparameters if hyperparameters.get('differentiable_hps_as_style', True) else None)

DataLoader = get_batch_to_dataloader(get_batch)

def draw_random_style(dl, device):
    (hp_embedding, _, _), _, _ = next(iter(dl))
    return hp_embedding.to(device)[0:1, :]

def merge_style_with_info(diff_hparams_keys, diff_hparams_f, style, transform=True):
    params = dict(zip(diff_hparams_keys, zip(diff_hparams_f, style.detach().cpu().numpy().tolist()[0])))
    def t(v):
        if transform:
            return v[0][1](v[1])
        else:
            return v[1]
    return {k : t(v) for k, v in params.items()}



def replace_differentiable_distributions(config):
    import ConfigSpace.hyperparameters as CSH
    diff_config = config['differentiable_hyperparameters']
    for name, diff_hp_dict in diff_config.items():
        distribution = diff_hp_dict['distribution']
        if distribution == 'uniform':
            diff_hp_dict['sample'] = CSH.UniformFloatHyperparameter(name, diff_hp_dict['min'], diff_hp_dict['max'])
        elif distribution == 'meta_beta':
            diff_hp_dict['k'] = CSH.UniformFloatHyperparameter(name+'_k', diff_hp_dict['min'], diff_hp_dict['max'])
            diff_hp_dict['b'] = CSH.UniformFloatHyperparameter(name+'_b', diff_hp_dict['min'], diff_hp_dict['max'])
        elif distribution == 'meta_gamma':
            diff_hp_dict['alpha'] = CSH.UniformFloatHyperparameter(name+'_k', 0.0, math.log(diff_hp_dict['max_alpha']))
            diff_hp_dict['scale'] = CSH.UniformFloatHyperparameter(name+'_b', 0.0, diff_hp_dict['max_scale'])
        elif distribution == 'meta_choice':
            for i in range(1, len(diff_hp_dict['choice_values'])):
                diff_hp_dict[f'choice_{i}_weight'] = CSH.UniformFloatHyperparameter(name+f'choice_{i}_weight', -3.0, 5.0)
        elif distribution == 'meta_choice_mixed':
            for i in range(1, len(diff_hp_dict['choice_values'])):
                diff_hp_dict[f'choice_{i}_weight'] = CSH.UniformFloatHyperparameter(name+f'choice_{i}_weight', -3.0, 5.0)
        elif distribution == 'meta_trunc_norm_log_scaled':
            diff_hp_dict['log_mean'] = CSH.UniformFloatHyperparameter(name+'_log_mean', math.log(diff_hp_dict['min_mean']), math.log(diff_hp_dict['max_mean']))
            min_std = diff_hp_dict['min_std'] if 'min_std' in diff_hp_dict else 0.1
            max_std = diff_hp_dict['max_std'] if 'max_std' in diff_hp_dict else 1.0
            diff_hp_dict['log_std'] = CSH.UniformFloatHyperparameter(name+'_log_std', math.log(min_std), math.log(max_std))
        else:
            raise ValueError(f'Unknown distribution {distribution}')

