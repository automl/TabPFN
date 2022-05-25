import random

import torch

from utils import set_locals_in_self
from .prior import PriorDataLoader
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.stats as stats
import math

def get_batch_to_dataloader(get_batch_method_):
    class DL(PriorDataLoader):
        get_batch_method = get_batch_method_

        # Caution, you might need to set self.num_features manually if it is not part of the args.
        def __init__(self, num_steps, fuse_x_y=False, **get_batch_kwargs):
            set_locals_in_self(locals())
            # The stuff outside the or is set as class attribute before instantiation.
            self.num_features = get_batch_kwargs.get('num_features') or self.num_features
            self.num_outputs = get_batch_kwargs.get('num_outputs') or self.num_outputs
            print('DataLoader.__dict__', self.__dict__)

        @staticmethod
        def gbm(*args, fuse_x_y=True, **kwargs):
            dynamic_seq_len = callable(kwargs['seq_len'])
            kwargs['seq_len'] = kwargs['seq_len']() if dynamic_seq_len else kwargs['seq_len']
            # Scales the batch size dynamically with the power of 'dynamic_batch_size'.
            # A transformer with quadratic memory usage in the seq len would need a power of 2 to keep memory constant.
            if dynamic_seq_len and 'dynamic_batch_size' in kwargs and kwargs['dynamic_batch_size'] > 0:
                kwargs['batch_size'] = kwargs['batch_size'] * math.floor(math.pow(kwargs['seq_len_maximum'], kwargs['dynamic_batch_size']) / math.pow(kwargs['seq_len'], kwargs['dynamic_batch_size']))
            batch = get_batch_method_(*args, **kwargs)
            x, y, target_y, style = batch if len(batch) == 4 else (batch[0], batch[1], batch[2], None)
            if fuse_x_y:
                return torch.cat([x, torch.cat([torch.zeros_like(y[:1]), y[:-1]], 0).unsqueeze(-1).float()],
                                 -1), target_y
            else:
                return (style, x, y), target_y

        def __len__(self):
            return self.num_steps

        def __iter__(self):
            return iter(self.gbm(**self.get_batch_kwargs, fuse_x_y=self.fuse_x_y) for _ in range(self.num_steps))


    return DL

import seaborn as sns
def plot_features(data, targets, fig=None):
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()
    #data = np.concatenate([data, data[:, -1:]], -1)
    #df = pd.DataFrame(data, columns=list(range(0, data.shape[1])))
    #g = sns.pairplot(df, hue=data.shape[1]-1, palette="Set2", diag_kind="kde", height=2.5)
    #plt.legend([], [], frameon=False)
    #g._legend.remove()
    #g = sns.PairGrid(df, hue=data.shape[1]-1)
    #g.map_diag(sns.histplot)
    #g.map_offdiag(sns.scatterplot)
    #g._legend.remove()

    fig2 = fig if fig else plt.figure(figsize=(8, 8))
    spec2 = gridspec.GridSpec(ncols=data.shape[1], nrows=data.shape[1], figure=fig2)
    for d in range(0, data.shape[1]):
        for d2 in range(0, data.shape[1]):
            sub_ax = fig2.add_subplot(spec2[d, d2])
            if d == d2:
                sns.kdeplot(data[:, d],hue=targets[:],ax=sub_ax,legend=False, palette="deep")
                sub_ax.set(ylabel=None)
            else:
                sns.scatterplot(x=data[:, d], y=data[:, d2],
                           hue=targets[:],legend=False, palette="deep")
                #plt.scatter(data[:, d], data[:, d2],
                #               c=targets[:])
            sub_ax.get_xaxis().set_ticks([])
            sub_ax.get_yaxis().set_ticks([])
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    fig2.show()


def plot_prior(prior):
    s = np.array([prior() for _ in range(0, 1000)])
    count, bins, ignored = plt.hist(s, 50, density=True)
    print(s.min())
    plt.show()

trunc_norm_sampler_f = lambda mu, sigma : lambda: stats.truncnorm((0 - mu) / sigma, (1000000 - mu) / sigma, loc=mu, scale=sigma).rvs(1)[0]
beta_sampler_f = lambda a, b : lambda : np.random.beta(a, b)
gamma_sampler_f = lambda a, b : lambda : np.random.gamma(a, b)
uniform_sampler_f = lambda a, b : lambda : np.random.uniform(a, b)
uniform_int_sampler_f = lambda a, b : lambda : round(np.random.uniform(a, b))
def zipf_sampler_f(a, b, c):
    x = np.arange(b, c)
    weights = x ** (-a)
    weights /= weights.sum()
    return lambda : stats.rv_discrete(name='bounded_zipf', values=(x, weights)).rvs(1)
scaled_beta_sampler_f = lambda a, b, scale, minimum : lambda : minimum + round(beta_sampler_f(a, b)() * (scale - minimum))


def normalize_by_used_features_f(x, num_features_used, num_features, normalize_with_sqrt=False):
    if normalize_with_sqrt:
        return x / (num_features_used / num_features)**(1 / 2)
    return x / (num_features_used / num_features)


def order_by_y(x, y):
    order = torch.argsort(y if random.randint(0, 1) else -y, dim=0)[:, 0, 0]
    order = order.reshape(2, -1).transpose(0, 1).reshape(-1)#.reshape(seq_len)
    x = x[order]  # .reshape(2, -1).transpose(0, 1).reshape(-1).flip([0]).reshape(seq_len, 1, -1)
    y = y[order]  # .reshape(2, -1).transpose(0, 1).reshape(-1).reshape(seq_len, 1, -1)

    return x, y

def randomize_classes(x, num_classes):
    classes = torch.arange(0, num_classes, device=x.device)
    random_classes = torch.randperm(num_classes, device=x.device).type(x.type())
    x = ((x.unsqueeze(-1) == classes) * random_classes).sum(-1)
    return x


class CategoricalActivation(nn.Module):
    def __init__(self, categorical_p=0.1, ordered_p=0.7
                 , keep_activation_size=False
                 , num_classes_sampler=zipf_sampler_f(0.8, 1, 10)):
        self.categorical_p = categorical_p
        self.ordered_p = ordered_p
        self.keep_activation_size = keep_activation_size
        self.num_classes_sampler = num_classes_sampler

        super().__init__()

    def forward(self, x):
        # x shape: T, B, H

        x = nn.Softsign()(x)

        num_classes = self.num_classes_sampler()
        hid_strength = torch.abs(x).mean(0).unsqueeze(0) if self.keep_activation_size else None

        categorical_classes = torch.rand((x.shape[1], x.shape[2])) < self.categorical_p
        class_boundaries = torch.zeros((num_classes - 1, x.shape[1], x.shape[2]), device=x.device, dtype=x.dtype)
        # Sample a different index for each hidden dimension, but shared for all batches
        for b in range(x.shape[1]):
            for h in range(x.shape[2]):
                ind = torch.randint(0, x.shape[0], (num_classes - 1,))
                class_boundaries[:, b, h] = x[ind, b, h]

        for b in range(x.shape[1]):
            x_rel = x[:, b, categorical_classes[b]]
            boundaries_rel = class_boundaries[:, b, categorical_classes[b]].unsqueeze(1)
            x[:, b, categorical_classes[b]] = (x_rel > boundaries_rel).sum(dim=0).float() - num_classes / 2

        ordered_classes = torch.rand((x.shape[1],x.shape[2])) < self.ordered_p
        ordered_classes = torch.logical_and(ordered_classes, categorical_classes)
        x[:, ordered_classes] = randomize_classes(x[:, ordered_classes], num_classes)

        x = x * hid_strength if self.keep_activation_size else x

        return x
