import unittest

import numpy as np
import torch
from tabpfn.scripts.transformer_prediction_interface import TabPFNClassifier
from tabpfn.datasets import load_openml_list, open_cc_dids, open_cc_valid_dids, test_dids_classification
from torch import nn
import random

class TestLoadModuleOnlyInference(unittest.TestCase):
    def test_main(self):

        max_samples = 10000
        bptt = 10000
        cc_test_datasets_multiclass, cc_test_datasets_multiclass_df = load_openml_list(open_cc_dids, multiclass=True,
                                                                                       shuffled=True,
                                                                                       filter_for_nan=False,
                                                                                       max_samples=max_samples,
                                                                                       num_feats=100,
                                                                                       return_capped=True)


        classifier_with_only_Inference = TabPFNClassifier(device='cpu', only_inference=True)
        classifier_normal = TabPFNClassifier(device='cpu', only_inference=False)
        classifier.fit(train_xs, train_ys)
        prediction_ = classifier.predict_proba(test_xs)