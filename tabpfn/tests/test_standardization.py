from tabpfn.utils import remove_outliers, torch_nanmean, torch_nanstd
from tabpfn.datasets import load_openml_list, open_cc_dids

import torch
import unittest
import numpy as np

def old_torch_nanmean(x, axis=0, return_nanshare=False):
    num = torch.where(torch.isnan(x), torch.full_like(x, 0), torch.full_like(x, 1)).sum(axis=axis)
    value = torch.where(torch.isnan(x), torch.full_like(x, 0), x).sum(axis=axis)
    if return_nanshare:
        return value / num, 1.-num/x.shape[axis]
    return value / num

def old_torch_nanstd(x, axis=0):
    num = torch.where(torch.isnan(x), torch.full_like(x, 0), torch.full_like(x, 1)).sum(axis=axis)
    value = torch.where(torch.isnan(x), torch.full_like(x, 0), x).sum(axis=axis)
    mean = value / num
    mean_broadcast = torch.repeat_interleave(mean.unsqueeze(axis), x.shape[axis], dim=axis)
    return torch.sqrt(torch.sum(torch.where(torch.isnan(x), torch.full_like(x, 0), torch.square(mean_broadcast - x)), dim=axis)/ (num - 1))
    #return torch.sqrt(torch.nansum(torch.square(mean_broadcast - x), dim=axis, dtype=torch.float64) / (num - 1))

def old_remove_outliers(X, n_sigma=4, normalize_positions=-1):
    # Expects T, B, H
    assert len(X.shape) == 3, "X must be T,B,H"
    data = X if normalize_positions == -1 else X[:normalize_positions]
    data_clean = data[:].clone()

    data_mean, data_std = old_torch_nanmean(data, axis=0), old_torch_nanstd(data, axis=0)
    cut_off = data_std * n_sigma
    lower, upper = data_mean - cut_off, data_mean + cut_off

    data_clean[torch.logical_or(data_clean > upper, data_clean < lower)] = np.nan

    data_mean, data_std = old_torch_nanmean(data_clean, axis=0), old_torch_nanstd(data_clean, axis=0)
    cut_off = data_std * n_sigma
    lower, upper = data_mean - cut_off, data_mean + cut_off

    X = torch.maximum(-torch.log(1+torch.abs(X)) + lower, X)
    X = torch.minimum(torch.log(1+torch.abs(X)) + upper, X)
    return X

class TestNewRemoveOutliers(unittest.TestCase):
    def test_main(self):
        test_datasets, cc_test_datasets_multiclass_df = load_openml_list(open_cc_dids)

        for dataset in test_datasets:
            xs = dataset[1].unsqueeze(1)

            xs_new = remove_outliers(xs)
            xs_old = old_remove_outliers(xs)

            xs_new = xs_new.squeeze(1)
            xs_old = xs_old.squeeze(1)

            number_of_samples, number_of_classes = xs_old.shape

            for number in range(number_of_samples):
                for class_nr in range(number_of_classes):
                    if torch.isnan(xs_new[number][class_nr]) and torch.isnan(xs_old[number][class_nr]):
                        continue
                    if float(xs_new[number][class_nr]) != float(xs_old[number][class_nr]):
                        print(float(xs_new[number][class_nr]) - float(xs_old[number][class_nr]))
                    # checks that every class probability has difference of at most
                    self.assertEqual(float(xs_new[number][class_nr]), float(xs_old[number][class_nr]))